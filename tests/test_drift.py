"""Known ground-truth return cycles, bias drift, slow motion, and recentering."""

import math
import unittest

import numpy as np

from imu_follow import ForearmFollower
from imu_hardware import IMUSample, RealIMU, gravity_in_body, multiply
from openarm_model import Control


def rotation(axis, degrees):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    half = math.radians(degrees) / 2
    return np.r_[math.cos(half), axis * math.sin(half)]


def angle_error(a, b):
    return math.degrees(2 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def calibrated():
    e = RealIMU()
    bias = np.radians([1.1, -1.2, 5.3])
    for i in range(100):
        e.update(IMUSample(i * 0.02, np.array([0.0, 0.0, 9.80665]), bias, 26))
    assert e.calibrated
    return e, bias


class DriftChecks(unittest.TestCase):
    def test_slow_tilt_is_rejected_during_calibration(self):
        e = RealIMU()
        for i in range(150):
            q = rotation([0, -1, 0], i * 0.02)
            e.update(
                IMUSample(i * 0.02, gravity_in_body(q) * 9.80665, np.radians([1.1, -2.2, 5.3]), 26)
            )
        self.assertFalse(e.calibrated)

    def test_temperature_bias_change_at_rest_converges(self):
        e, bias = calibrated()
        rng = np.random.default_rng(11)
        yaw = []
        for i in range(3000):
            added_bias = np.radians([0.0, 0.0, 0.12])
            e.update(
                IMUSample(
                    e.last_t + 0.02,
                    np.array([0.0, 0.0, 9.80665]) + rng.normal(0, 0.012, 3),
                    bias + added_bias + rng.normal(0, math.radians(0.04), 3),
                    28,
                )
            )
            q = e.q
            yaw.append(
                math.degrees(
                    math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                )
            )
        self.assertTrue(e.stationary)
        self.assertLess(abs(math.degrees(e.bias[2] - bias[2]) - 0.12), 0.025)
        self.assertLess(
            abs(yaw[-1]), 1.0, "Residual bias should not integrate into 7+ degrees of drift"
        )
        self.assertLess(max(yaw[-500:]) - min(yaw[-500:]), 0.1)

    def test_one_degree_per_second_yaw_remains_visible(self):
        e, bias = calibrated()
        for _ in range(500):
            e.update(
                IMUSample(
                    e.last_t + 0.02, np.array([0.0, 0.0, 9.80665]), bias + np.radians([0, 0, 1]), 26
                )
            )
        self.assertFalse(e.stationary, "Slow intended yaw must not be learned as zero rate")
        self.assertLess(angle_error(e.q, rotation([0, 0, 1], 10)), 0.25)

    def test_repeated_return_cycles_with_timestamps(self):
        e, bias = calibrated()
        t = 0.0
        for i in range(2500):
            # Variable arrival/sample intervals spanning missing 50 Hz ticks.
            dt = 0.04 if i % 13 == 0 else 0.02
            previous = t
            t += dt
            degrees = 35 * math.sin(2 * math.pi * 0.25 * t)
            previous_degrees = 35 * math.sin(2 * math.pi * 0.25 * previous)
            q = rotation([0, -1, 0], degrees)
            omega = np.radians([0, -(degrees - previous_degrees) / dt, 0])
            e.update(IMUSample(e.last_t + dt, gravity_in_body(q) * 9.80665, bias + omega, 26))
        for _ in range(150):
            e.update(IMUSample(e.last_t + 0.02, gravity_in_body(q) * 9.80665, bias, 26))
        self.assertLess(angle_error(e.q, q), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(e.q)), 1.0)

    def test_recenter_keeps_original_anchor_and_taught_axes(self):
        specs = [
            Control(f"J{i}", "", -90 if i != 4 else 0, 140, 90 if i == 4 else 0)
            for i in range(1, 8)
        ]
        specs.append(Control("Gripper", "", 0, 100, 0, True))
        f = ForearmFollower(specs)
        f.update(rotation([0, 1, 0], 0), 0)
        for i in range(1, 101):
            f.update(rotation([0, -1, 0], 30 * i / 100), i * 0.02)
        original_anchor = f.base.copy()
        taught = f.forward.copy()
        # An accumulated yaw offset at the original physical pose.
        q_offset = rotation([0, 0, 1], 100)
        for i in range(100):
            f.update(q_offset, f.last_t + 0.02)
        self.assertTrue(f.limited)
        self.assertTrue(f.recenter(q_offset, f.last_t))
        np.testing.assert_allclose(f.commands, original_anchor)
        np.testing.assert_allclose(f.forward, taught)
        for i in range(100):
            f.update(multiply(q_offset, rotation([0, -1, 0], 30)), f.last_t + 0.02)
        self.assertAlmostEqual(f.commands[3], 119.6, delta=0.05)
        self.assertFalse(f.limited)
        np.testing.assert_allclose(f.base, original_anchor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
