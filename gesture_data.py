"""Strict sensor-only recording format used at the student boundary."""

import hashlib
from pathlib import Path

import numpy as np

from gesture_model import STRIDE_SAMPLES, WINDOW_SAMPLES, window_features


def load_imu(path):
    """Reject files containing privileged fields, even if the model ignores them."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"time", "signal"}:
            raise ValueError("Student input must contain only time and signal")
        t, signal = data["time"], data["signal"]
    if (
        t.ndim != 1
        or len(t) < WINDOW_SAMPLES
        or signal.shape != (len(t), 6)
        or not np.isfinite(t).all()
        or not np.isfinite(signal).all()
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("Invalid student IMU recording")
    if not np.allclose(np.diff(t), 0.02, atol=1e-5):
        raise ValueError("Offline student recordings must be sampled at 50 Hz")
    return t, signal


def recording_features(t, signal):
    ends = np.arange(WINDOW_SAMPLES - 1, len(t), STRIDE_SAMPLES)
    windows = np.array([signal[i - WINDOW_SAMPLES + 1 : i + 1] for i in ends])
    return ends, window_features(windows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
