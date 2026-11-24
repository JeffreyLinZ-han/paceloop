"""Inference timing, event gating, integration, and simulated sensor checks."""

import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np

from gesture_model import (
    BiteEventGate,
    GestureRecognizer,
    TinyGestureModel,
    window_features,
)
from imu_hardware import IMUSample
from run_twin import BiteTimer, PaceLoopApp
from tools.simulate_gestures import GestureSimulator

ROOT = Path(__file__).resolve().parents[1]


class GestureChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TinyGestureModel.load(ROOT / "models" / "gesture_sim_v1.npz")

    def test_archive_round_trip_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            self.model.save(path)
            restored = TinyGestureModel.load(path)
            np.testing.assert_array_equal(restored.w1, self.model.w1)
            params = [a.copy() for a in self.model.arrays]
            params[1][0] = 0
            with self.assertRaises(ValueError):
                TinyGestureModel(*params, self.model.metadata)
        with self.assertRaises(ValueError):
            window_features(np.zeros((50, 6)))
        bad = np.zeros((120, 6))
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            window_features(bad)

    def test_stream_matches_offline_window_with_uneven_packets(self):
        t = np.arange(160) / 50
        signal = np.column_stack([np.sin(t * (i + 1)) for i in range(6)])
        signal[:, 2] += 9.80665
        recognizer = GestureRecognizer(self.model)
        # Packet delivery cadence does not replace the device timestamps.
        for start in range(0, len(t), 7):
            for i in range(start, min(start + 7, len(t))):
                recognizer.update(float(t[i]), signal[i, :3], signal[i, 3:])
        expected = self.model.probabilities(window_features(signal[-120:]))
        np.testing.assert_allclose(recognizer.scores, expected, atol=2e-6)
        np.testing.assert_allclose(recognizer.scores.sum(), 1, atol=1e-6)

    def test_stale_restart_duplicate_and_nonfinite_data(self):
        recognizer = GestureRecognizer(self.model)
        for i in range(130):
            recognizer.update(i / 50, [0, 0, 9.8], [0, 0, 0])
        count = recognizer.count
        recognizer.update(129 / 50, [0, 0, 9.8], [0, 0, 0])
        self.assertEqual(recognizer.count, count)
        recognizer.update(8, [0, 0, 9.8], [0, 0, 0])
        self.assertIsNone(recognizer.scores)
        self.assertEqual(recognizer.count, 1)
        recognizer.update(8.02, [0, 0, 9.8], [0, 0, 0], restart=True)
        self.assertEqual(recognizer.count, 1)
        recognizer.update(8.04, [np.nan, 0, 9.8], [0, 0, 0])
        self.assertEqual(recognizer.count, 0)

    def test_event_gate_requires_confidence_and_release(self):
        gate = BiteEventGate(0.7)
        bite, other = [0.02, 0.94, 0.02, 0.02], [0.94, 0.02, 0.02, 0.02]
        self.assertFalse(gate.update(0, bite))
        self.assertFalse(gate.update(0.2, other))
        events = [t for t in np.arange(0.4, 3, 0.2) if gate.update(float(t), bite)]
        self.assertEqual(len(events), 1)
        for t in np.arange(3, 4, 0.2):
            self.assertFalse(gate.update(float(t), other))
        self.assertFalse(gate.update(4, bite))
        self.assertTrue(gate.update(4.2, bite))

    def test_learned_detection_replaces_tilt_proxy_and_drives_bridge(self):
        app = object.__new__(PaceLoopApp)
        app.bridge = SimpleNamespace(live=True, target_s=5, alert=Mock(), drain=Mock())
        app.real_imu = SimpleNamespace(
            segment=0,
            calibrated=True,
            update=Mock(),
            tilt=80,
            corrected_gyro=np.ones(3),
            q=[1, 0, 0, 0],
            status="LIVE",
        )
        app.was_live = True
        app.follow_enabled = False
        app.gesture = Mock()
        app.gesture.update.return_value = False
        app.history = [deque(maxlen=200) for _ in range(6)]
        app.timer = BiteTimer()
        app._log_sample = Mock()
        sample = IMUSample(3, np.array([0, 0, 9.8]), np.ones(3), 25)
        app.bridge.drain.return_value = [sample]
        app._process_hardware(99, [])
        self.assertEqual(app.timer.bite_count, 0, "Tilt proxy must not bypass the model")
        app.gesture.update.return_value = True
        app._process_hardware(100, [])
        sample.t = 5
        app._process_hardware(102, [])
        self.assertEqual(app.timer.bite_count, 2)
        app.bridge.alert.assert_called_once()

    def test_inverse_accelerometer_matches_kinematic_finite_difference(self):
        simulator = GestureSimulator()
        model, data = simulator.model, simulator.data
        base = np.deg2rad([15, 20, -10, 110, 5, 3, 2])
        amplitude = np.array([0.1, 0.04, 0.03, 0.2, 0.04, 0.02, 0.01])
        t, dt = 0.6, 0.0001
        positions = []
        for stamp in (t - dt, t, t + dt):
            data.qpos[simulator.qids] = base + amplitude * np.sin(stamp)
            data.qvel[:] = 0
            data.qacc[:] = 0
            data.qvel[simulator.vids] = amplitude * np.cos(stamp)
            data.qacc[simulator.vids] = -amplitude * np.sin(stamp)
            mujoco.mj_inverse(model, data)
            positions.append(data.site("imu").xpos.copy())
            if stamp == t:
                rotation = data.site("imu").xmat.reshape(3, 3).copy()
                measured = data.sensor("imu_accel").data.copy()
        world_acc = (positions[0] - 2 * positions[1] + positions[2]) / dt**2
        expected = rotation.T @ (world_acc - model.opt.gravity)
        np.testing.assert_allclose(measured, expected, atol=2e-5)

    def test_training_split_and_model_report(self):
        report = json.loads((ROOT / "models" / "gesture_sim_v1.metrics.json").read_text())
        groups = report["split_sessions"]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(set(groups[a]) & set(groups[b]))
        self.assertFalse(report["real_world_validated"])
        self.assertEqual(report["source"], "simulation_only")


if __name__ == "__main__":
    unittest.main(verbosity=2)
