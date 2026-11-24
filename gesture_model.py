"""Small NumPy gesture classifier and timestamped streaming inference."""

import json
from collections import deque
from pathlib import Path

import numpy as np

LABELS = ("other", "bite", "scratch_head", "touch_face")
SAMPLE_HZ = 50
WINDOW_SAMPLES = 120
STRIDE_SAMPLES = 10
FEATURE_VERSION = 1
DEFAULT_MODEL_FILE = "gesture_student_v2.npz"


def window_features(windows):
    """Pool 2.4 seconds of six SI-unit channels, retaining their time order."""
    x = np.asarray(windows, dtype=np.float32)
    if x.shape[-2:] != (WINDOW_SAMPLES, 6) or not np.isfinite(x).all():
        raise ValueError("Expected finite (..., 120, 6) IMU windows")
    x = x / np.array([9.80665] * 3 + [3.0] * 3, dtype=np.float32)
    magnitudes = np.stack(
        (np.linalg.norm(x[..., :3], axis=-1), np.linalg.norm(x[..., 3:], axis=-1)), axis=-1
    )
    x = np.concatenate((x, magnitudes), axis=-1)
    pooled = x.reshape(*x.shape[:-2], 24, 5, 8).mean(axis=-2)
    return np.concatenate(
        (pooled.reshape(*x.shape[:-2], 192), x.std(axis=-2), np.ptp(x, axis=-2)), axis=-1
    )


class TinyGestureModel:
    def __init__(self, mean, scale, w1, b1, w2, b2, metadata):
        self.mean, self.scale, self.w1, self.b1, self.w2, self.b2 = [
            np.asarray(v, dtype=np.float32) for v in (mean, scale, w1, b1, w2, b2)
        ]
        self.metadata = metadata
        hidden = self.b1.size
        expected = ((208,), (208,), (208, hidden), (hidden,), (hidden, 4), (4,))
        if [v.shape for v in self.arrays] != list(expected):
            raise ValueError("Invalid gesture network dimensions")
        if not all(np.isfinite(v).all() for v in self.arrays) or np.any(self.scale <= 0):
            raise ValueError("Invalid gesture network values")
        if (
            metadata.get("feature_version") != FEATURE_VERSION
            or metadata.get("labels") != list(LABELS)
            or metadata.get("sample_hz") != SAMPLE_HZ
            or metadata.get("window_samples") != WINDOW_SAMPLES
        ):
            raise ValueError("Incompatible gesture model")
        self.threshold = float(metadata.get("threshold", 0.8))
        if not 0.5 <= self.threshold <= 0.99:
            raise ValueError("Invalid detection threshold")

    @property
    def arrays(self):
        return self.mean, self.scale, self.w1, self.b1, self.w2, self.b2

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            return cls(*(data[k] for k in ("mean", "scale", "w1", "b1", "w2", "b2")), metadata)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                **dict(zip(("mean", "scale", "w1", "b1", "w2", "b2"), self.arrays)),
                metadata=json.dumps(self.metadata),
            )

    def probabilities(self, features):
        x = np.clip((np.asarray(features) - self.mean) / self.scale, -8, 8)
        hidden = np.maximum(x @ self.w1 + self.b1, 0)
        logits = hidden @ self.w2 + self.b2
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)


class BiteEventGate:
    """Require sustained confidence and a release before another event."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.armed = True
        self.high_since = self.low_since = self.last_event = None

    def update(self, t, probabilities):
        score = float(probabilities[1])
        high = int(np.argmax(probabilities)) == 1 and score >= self.threshold
        if score < 0.35:
            self.low_since = t if self.low_since is None else self.low_since
            if t - self.low_since >= 0.6 - 1e-6:
                self.armed = True
        else:
            self.low_since = None
        if not high:
            self.high_since = None
            return False
        self.high_since = t if self.high_since is None else self.high_since
        if (
            self.armed
            and t - self.high_since >= 0.2 - 1e-6
            and (self.last_event is None or t - self.last_event >= 1.2)
        ):
            self.armed = False
            self.last_event = t
            return True
        return False


class GestureRecognizer:
    """Resample device time to 50 Hz; gaps/restarts discard incomplete windows."""

    def __init__(self, model):
        self.model = model
        self.gate = BiteEventGate(model.threshold)
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=WINDOW_SAMPLES)
        self.previous = None
        self.next_t = None
        self.count = 0
        self.scores = None
        self.prediction_count = 0
        self.prediction_time = None
        self.label = "warming_up"
        self.gate.reset()

    def update(self, t, accel, gyro, restart=False):
        values = np.concatenate((accel, gyro)).astype(np.float32)
        if values.shape != (6,) or not np.isfinite(values).all() or not np.isfinite(t):
            self.reset()
            return False
        if restart or (
            self.previous is not None and (t < self.previous[0] or t - self.previous[0] > 0.25)
        ):
            self.reset()
        if self.previous is None:
            self.previous = (t, values)
            self.next_t = t
        old_t, old_values = self.previous
        if t == old_t and self.count:
            return False
        event = False
        while self.next_t <= t + 1e-8:
            fraction = np.clip((self.next_t - old_t) / max(t - old_t, 1e-8), 0, 1)
            self.samples.append(old_values + fraction * (values - old_values))
            self.count += 1
            if (
                len(self.samples) == WINDOW_SAMPLES
                and (self.count - WINDOW_SAMPLES) % STRIDE_SAMPLES == 0
            ):
                self.scores = self.model.probabilities(window_features(np.asarray(self.samples)))
                self.prediction_count += 1
                self.prediction_time = self.next_t
                self.label = LABELS[int(np.argmax(self.scores))]
                event |= self.gate.update(self.next_t, self.scores)
            self.next_t += 1.0 / SAMPLE_HZ
        self.previous = (t, values)
        return event
