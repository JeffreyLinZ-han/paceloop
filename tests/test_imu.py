"""Offline transport, calibration and hardware/simulation separation checks."""

import math
import time
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from imu_hardware import ArduinoBridge, IMUSample, RealIMU
from run_twin import BiteTimer, PaceLoopApp


def frame(us, accel=(0, 0, 9.80665), gyro=(0, 0, 0)):
    return f"IMU,{us}," + ",".join(str(v) for v in (*accel, *gyro, 25))


def calibrate(estimator, accel=(0, 0, 9.80665), gyro=(0.01, -0.02, 0.09)):
    for i in range(100):
        estimator.update(IMUSample(i * 0.02, np.array(accel), np.array(gyro), 25))
    return 1.98


class HardwareChecks(unittest.TestCase):
    def test_protocol_identity_and_malformed_frames(self):
        bridge = ArduinoBridge(None)
        bridge.accept_line("IMU_STATUS,OK,0x68,0x70")
        bridge.accept_line("MODE,4000")
        bridge.accept_line(frame(20000))
        self.assertEqual(bridge.chip_name, "MPU6500-compatible")
        self.assertEqual(bridge.mode_name, "CUSTOM")
        self.assertEqual(bridge.target_s, 4)
        for bad in (
            "IMU,3,1",
            frame(-1),
            frame(2**32),
            frame(40000, (float("nan"), 0, 0)),
            frame(40000, gyro=(10, 0, 0)),
        ):
            bridge.accept_line(bad)
        self.assertEqual(bridge.invalid_frames, 5)
        self.assertEqual(len(bridge.drain()), 1)

    def test_wrap_duplicate_and_reboot(self):
        bridge = ArduinoBridge(None)
        for stamp in (2**32 - 10000, 10000, 10000, 30000, 500):
            bridge.accept_line(frame(stamp))
        samples = bridge.drain()
        self.assertEqual(len(samples), 4)
        np.testing.assert_allclose(np.diff([s.t for s in samples]), 0.02, atol=1e-9)
        self.assertFalse(samples[1].restart)
        self.assertTrue(samples[-1].restart)

    def test_partial_serial_and_stale(self):
        bridge = ArduinoBridge(None)
        transport = SimpleNamespace(in_waiting=16, read=Mock(return_value=b"MODE,15000\nIMU,2"))
        bridge.ser = transport
        bridge.poll()
        self.assertEqual(bridge.mode_name, "SLOW")
        self.assertFalse(bridge.live)
        transport.read.return_value = b"0000,0,0,9.80665,0,0,0,25\n"
        bridge.poll()
        self.assertTrue(bridge.live)
        bridge.last_rx = time.monotonic() - 0.6
        self.assertFalse(bridge.live)
        bridge.accept_line("IMU_STATUS,READ_ERROR")
        self.assertEqual(bridge.drain(), [])

    def test_calibration_and_rotation(self):
        estimator = RealIMU()
        calibrate(estimator)
        self.assertTrue(estimator.calibrated)
        np.testing.assert_allclose(estimator.bias, [0.01, -0.02, 0.09])
        self.assertLess(estimator.tilt, 0.01)
        # Known 40-degree rotation around body X in one second, with gravity.
        events = []
        timer = BiteTimer()
        for i in range(1, 51):
            angle = math.radians(40 * i / 50)
            acc = np.array([0, math.sin(angle), math.cos(angle)]) * 9.80665
            gyro = estimator.bias + np.array([math.radians(40), 0, 0])
            sample = IMUSample(1.98 + i * 0.02, acc, gyro, 25)
            estimator.update(sample)
            event = timer.step(sample.t, estimator.tilt, estimator.corrected_gyro, 5)
            if event:
                events.append(event)
        self.assertAlmostEqual(estimator.tilt, 40, delta=1)
        self.assertEqual(events, ["FIRST BITE"])
        self.assertAlmostEqual(float(np.linalg.norm(estimator.q)), 1)

    def test_arbitrary_neutral_and_interruption(self):
        for acc in ((9.80665, 0, 0), (0, 0, -9.80665)):
            estimator = RealIMU()
            calibrate(estimator, acc)
            self.assertTrue(estimator.calibrated)
            self.assertLess(estimator.tilt, 0.01)
            segment = estimator.segment
            estimator.update(IMUSample(4, np.array(acc), np.zeros(3), 25))
            self.assertFalse(estimator.calibrated)
            self.assertGreater(estimator.segment, segment)

    def test_moving_sensor_does_not_calibrate(self):
        estimator = RealIMU()
        for i in range(200):
            estimator.update(
                IMUSample(i * 0.02, np.array([0, 0, 9.80665]), np.array([(-1) ** i, 0, 0]), 25)
            )
        self.assertFalse(estimator.calibrated)

    def test_no_simulation_fallback_and_loss_resets_timing(self):
        app = object.__new__(PaceLoopApp)
        app.bridge = ArduinoBridge(None)
        app.real_imu = RealIMU()
        app.timer = BiteTimer()
        app.timer.last_bite_time = 3
        app.was_live = True
        app.follow_enabled = False
        app.history = [deque([123]) for _ in range(6)]
        app._log_sample = Mock()
        app._process_hardware(1000, [])
        self.assertIsNone(app.timer.last_bite_time)
        self.assertFalse(app.real_imu.calibrated)
        self.assertTrue(all(not v for v in app.history))
        app._log_sample.assert_not_called()

    def test_mode_and_alert_protocol_retained(self):
        bridge = ArduinoBridge(None)
        bridge.ser = SimpleNamespace(write=Mock())
        bridge.alert()
        bridge.ser.write.assert_called_once_with(b"ALERT\n")
        for value, name in ((3000, "FAST"), (5000, "MEDIUM"), (15000, "SLOW")):
            bridge.accept_line(f"MODE,{value}")
            self.assertEqual(bridge.mode_name, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
