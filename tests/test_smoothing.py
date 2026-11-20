"""Check mid-movement reversal, joint bounds and derivative continuity."""

import unittest

import numpy as np

from openarm_model import SmoothControls, controls_for, load_model


class SmoothingTests(unittest.TestCase):
    def test_continuity_and_bounds(self):
        model, data = load_model()
        specs = controls_for(model)
        motion = SmoothControls(model, data, specs)
        dt = model.opt.timestep
        commands = [s.home for s in specs]
        previous_acc = motion.accel.copy()
        peak_v = peak_a = peak_j = 0.0
        for step in range(10000):
            t = step * dt
            if step % 10 == 0:
                # Reversals while moving, including an endpoint near the wrist's joint limit.
                commands[5] = 45 if t < 1.5 else -45 if t < 3 else 45 if t < 5 else 0
                before = np.r_[motion.position, motion.velocity, motion.accel].copy()
                motion.set_targets(commands)
                np.testing.assert_array_equal(
                    np.r_[motion.position, motion.velocity, motion.accel], before
                )
            motion.step(dt)
            peak_v = max(peak_v, float(np.max(np.abs(motion.velocity[:7]))))
            peak_a = max(peak_a, float(np.max(np.abs(motion.accel[:7]))))
            peak_j = max(peak_j, float(np.max(np.abs(motion.accel[:7] - previous_acc[:7]))) / dt)
            previous_acc[:] = motion.accel
            for i in range(7):
                lo, hi = model.actuator_ctrlrange[motion.ids[i]]
                assert lo - 1e-8 <= motion.position[i] <= hi + 1e-8
        assert peak_v <= motion.speed * 1.001
        assert peak_a <= motion.acceleration * 1.001
        assert peak_j <= motion.jerk * 1.001
        assert abs(motion.position[5]) < 1e-8
        assert abs(motion.velocity[5]) < 1e-8
        assert abs(motion.accel[5]) < 1e-8
