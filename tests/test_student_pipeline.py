"""The student accepts only IMU; independent teacher targets determine loss."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gesture_data import load_imu, sha256
from gesture_metrics import cross_entropy
from gesture_model import DEFAULT_MODEL_FILE
from tools.evaluate_gestures import evaluate
from tools.predict_gestures import predict
from tools.prepare_student_data import prepare

ROOT = Path(__file__).resolve().parents[1]


class StudentPipelineChecks(unittest.TestCase):
    def test_cross_entropy_has_known_values(self):
        self.assertAlmostEqual(cross_entropy(np.array([0, 2]), np.full((2, 4), 0.25)), math.log(4))
        self.assertAlmostEqual(cross_entropy(np.array([0, 2]), np.eye(4)[[0, 2]]), 0)
        self.assertGreater(
            cross_entropy(np.array([0]), np.array([[0.001, 0.997, 0.001, 0.001]])), 6
        )
        expected = (-math.log(0.8) - 3 * math.log(0.5)) / 4
        self.assertAlmostEqual(
            cross_entropy(
                np.array([0, 1]),
                np.array([[0.8, 0.1, 0.05, 0.05], [0.2, 0.5, 0.2, 0.1]]),
                [1, 3, 1, 1],
            ),
            expected,
        )
        with self.assertRaises(ValueError):
            cross_entropy(np.array([0]), np.array([[2, -1, 0, 0]]))

    def test_student_rejects_labels_and_joint_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "imu.npz"
            t, signal = np.arange(140) / 50, np.zeros((140, 6))
            for forbidden in ("labels", "joints", "action", "wrist"):
                np.savez(path, time=t, signal=signal, **{forbidden: [0]})
                with self.assertRaisesRegex(ValueError, "only time and signal"):
                    load_imu(path)
            np.savez(path, time=t, signal=signal)
            loaded_t, loaded_signal = load_imu(path)
            np.testing.assert_array_equal(loaded_t, t)
            np.testing.assert_array_equal(loaded_signal, signal)

    def test_prediction_precedes_teacher_scoring_and_is_invariant_to_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "recorded"
            source.mkdir()
            t = np.arange(160) / 50
            signal = np.column_stack([np.sin((i + 1) * t) for i in range(6)])
            signal[:, 2] += 9.80665
            np.savez(
                source / "known_bite.npz",
                time=t,
                signal=signal,
                labels=np.ones(len(t), dtype=int),
                joints=np.zeros((len(t), 7)),
                wrist=np.zeros((len(t), 3)),
                session=42,
                repetition=0,
                action=1,
                event_interval=[2.3, 3.2],
            )
            (source / "manifest.json").write_text(
                json.dumps(
                    dict(episodes=[dict(file="known_bite.npz", split="test", session=42, action=1)])
                )
            )
            data = root / "separated"
            prepared = prepare(source, data)
            filename = prepared["episodes"][0]["file"]
            self.assertNotIn("bite", filename)
            with np.load(data / "imu" / "test" / filename) as student:
                self.assertEqual(set(student.files), {"time", "signal"})
            model = ROOT / "models" / DEFAULT_MODEL_FILE
            model_hash = sha256(model)
            predict(model, data / "imu" / "test", root / "predictions1")
            before = evaluate(root / "predictions1", data / "teacher" / "test")
            teacher_path = data / "teacher" / "test" / filename
            with np.load(teacher_path) as f:
                teacher = {k: f[k] for k in f.files}
            teacher["labels"] = (teacher["labels"] + 1) % 4
            np.savez_compressed(teacher_path, **teacher)
            predict(model, data / "imu" / "test", root / "predictions2")
            with (
                np.load(root / "predictions1" / filename) as first,
                np.load(root / "predictions2" / filename) as second,
            ):
                np.testing.assert_array_equal(first["scores"], second["scores"])
                self.assertNotIn("labels", first.files)
            after = evaluate(root / "predictions2", data / "teacher" / "test")
            self.assertNotAlmostEqual(
                before["classification"]["cross_entropy"], after["classification"]["cross_entropy"]
            )
            self.assertEqual(
                sha256(model), model_hash, "Evaluation must not update student weights"
            )
            teacher["input_sha256"] = "different_recording"
            np.savez_compressed(teacher_path, **teacher)
            with self.assertRaisesRegex(ValueError, "different sensor recordings"):
                evaluate(root / "predictions2", data / "teacher" / "test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
