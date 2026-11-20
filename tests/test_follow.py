"""Known rotations, arbitrary mounting, tracking dynamics and recorded-IMU replay."""

import csv
import math
import unittest
from pathlib import Path

import mujoco
import numpy as np

from imu_follow import ForearmFollower, LiveControls
from imu_hardware import IMUSample, RealIMU, multiply
from openarm_model import controls_for, load_model


def rotation(axis, degrees):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    half = math.radians(degrees) / 2
    return np.r_[math.cos(half), axis * math.sin(half)]


class FollowChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data = load_model()
        cls.specs = controls_for(cls.model)

    def trained(self, mount=np.array([1.0, 0, 0, 0])):
        follower = ForearmFollower(self.specs)
        follower.update(mount, 0)
        for i in range(1, 51):
            follower.update(multiply(rotation([0, -1, 0], 30 * i / 50), mount), i * 0.02)
        return follower

    def test_mounting_independent_lift_heading_twist_and_neutral(self):
        for mount in (np.array([1.0, 0, 0, 0]), rotation([1, 2, -3], 123)):
            follower = self.trained(mount)
            for axis, degrees, expected in (
                ([0, -1, 0], 30, [30, 0, 0]),
                ([0, 0, 1], 25, [0, 25, 0]),
                ([1, 0, 0], 40, [0, 0, 40]),
                ([1, 0, 0], 0, [0, 0, 0]),
            ):
                q = multiply(rotation(axis, degrees), mount)
                for _ in range(80):
                    follower.update(q, follower.last_t + 0.02)
                np.testing.assert_allclose(follower.angles, expected, atol=0.41)
            np.testing.assert_allclose(follower.commands, [s.home for s in self.specs], atol=0.01)

    def test_teaching_hold_high_lift_and_joint_limits(self):
        f = ForearmFollower(self.specs)
        home = f.commands.copy()
        for i in range(20):
            f.update(rotation([0, 0, 1], i), i * 0.02)
        self.assertIsNone(f.forward, "Yaw alone must not teach a lift axis")
        np.testing.assert_equal(f.commands, home)
        f = self.trained()
        for i in range(1, 150):
            f.update(rotation([0, -1, 0], 70), f.last_t + 0.02)
        self.assertAlmostEqual(f.commands[3], 130, delta=0.01)
        self.assertAlmostEqual(f.commands[0], 29.6, delta=0.01)
        for i in range(1, 250):
            q = rotation([1, 0, 0], i)
            f.update(q if i % 2 else -q, f.last_t + 0.02)
            self.assertTrue(np.isfinite(f.commands).all())
            for s, value in zip(self.specs, f.commands):
                self.assertLessEqual(value, s.upper)
                self.assertGreaterEqual(value, s.lower)
        self.assertTrue(f.limited)
        self.assertGreater(f.commands[4], 70, "Quaternion sign/wrap must not reverse the wrist")

    def test_continuous_tracking_and_braking(self):
        m, d = self.model, mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        live = LiveControls(m, d, self.specs)
        targets = np.array([s.home for s in self.specs])
        previous_v = live.velocity.copy()
        maximum_error = 0.0
        for step in range(4000):
            t = step * 0.002
            if step % 10 == 0:
                targets[3] = 90 + 25 * math.sin(2 * math.pi * 0.3 * t)
                live.set_targets(targets)
            before = live.position.copy()
            live.step(0.002)
            self.assertLessEqual(np.max(np.abs(live.position - before)), live.speed * 0.002 + 1e-9)
            self.assertLessEqual(
                np.max(np.abs(live.velocity - previous_v)), live.acceleration * 0.002 + 1e-9
            )
            previous_v = live.velocity.copy()
            if t > 1:
                maximum_error = max(maximum_error, abs(math.degrees(live.position[3]) - targets[3]))
        self.assertLess(maximum_error, 4.0, f"Moving input lag too high: {maximum_error}")
        held = [math.degrees(x) if not s.gripper else 0 for s, x in zip(self.specs, live.position)]
        live.set_targets(held)
        for _ in range(500):
            live.step(0.002)
        self.assertLess(abs(math.degrees(live.position[3]) - held[3]), 0.01)

    def test_recorded_real_sensor_drives_physics(self):
        # Use an existing hardware CSV when available; never label replay live.
        path = Path(__file__).with_name("data") / "forearm_imu.csv"
        if not path.exists():
            self.skipTest("No recorded hardware session available")
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [row for _, row in zip(range(5000), reader) if row["imu_source"] == "hardware"]
        self.assertGreater(len(rows), 200)
        m = self.model
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        mujoco.mj_forward(m, d)
        left = d.body("openarm_left_link7").xpos.copy()
        imu = RealIMU()
        follower = ForearmFollower(self.specs)
        live = LiveControls(m, d, self.specs)
        actual, commanded, wrists = [], [], []
        for row in rows:
            sample = IMUSample(
                float(row["sensor_time_s"]),
                np.array([float(row[k]) for k in ("ax", "ay", "az")]),
                np.array([float(row[k]) for k in ("gx", "gy", "gz")]),
                float(row["temperature_C"]),
            )
            segment = imu.segment
            imu.update(sample)
            if imu.segment != segment:
                follower.reset(
                    [
                        math.degrees(v) if not s.gripper else 0
                        for s, v in zip(self.specs, live.position)
                    ]
                )
                live.set_targets(follower.commands)
            if imu.calibrated:
                live.set_targets(follower.update(imu.q, sample.t))
            for _ in range(10):
                live.step(0.002)
                mujoco.mj_step(m, d)
            actual.append([float(d.joint(f"openarm_right_joint{j}").qpos[0]) for j in range(1, 8)])
            commanded.append(live.position[:7].copy())
            wrists.append(d.sensor("wrist_pos").data.copy())
            self.assertTrue(np.isfinite(d.qpos).all())
        np.testing.assert_allclose(d.body("openarm_left_link7").xpos, left, atol=1e-12)
        ranges = np.ptp(np.degrees(actual), axis=0)
        self.assertGreater(ranges[3], 10.0, "Recorded forearm movement did not move the elbow")
        self.assertLess(np.max(np.abs(np.degrees(np.array(actual) - commanded))), 20.0)
        self.assertFalse(any(w.number for w in d.warning))


if __name__ == "__main__":
    unittest.main(verbosity=2)
