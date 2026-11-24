"""MPU6050/MPU6500 serial transport and VQF orientation estimation in SI units."""

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    from vqf import VQF
except ImportError:
    VQF = None

try:
    import serial
except ImportError:
    serial = None

MODE_NAMES = {3000: "FAST", 5000: "MEDIUM", 15000: "SLOW"}


@dataclass
class IMUSample:
    t: float
    accel: np.ndarray
    gyro: np.ndarray
    temperature: float
    restart: bool = False


@dataclass
class RuleState:
    t: float
    ready: bool
    count: int
    previous: float | None
    age: float
    result: str


class ArduinoBridge:
    def __init__(self, port, control_mode=None):
        if control_mode not in (None, "PC", "RULES"):
            raise ValueError("control_mode must be PC or RULES")
        self.ser = None
        self.operating_mode = "UNKNOWN"
        self.rule_state = None
        self.host_enabled = control_mode is not None
        self._requested_control = control_mode
        self._last_ping = 0.0
        self.target_ms = 5000
        self.rx_buffer = bytearray()
        self.status = "SIMULATION ONLY"
        self.imu_status = "WAITING FOR IMU"
        self.chip_name = "IMU"
        self.samples = deque(maxlen=500)
        self.latest = None
        self.last_rx = None
        self.last_us = None
        self.time_offset = 0
        self.invalid_frames = 0
        self.info = ""
        if not port:
            return
        if serial is None:
            self.status = "PYSERIAL MISSING"
            return
        try:
            self.ser = serial.Serial(port, 115200, timeout=0, write_timeout=0.5)
            time.sleep(2)
            self.ser.reset_input_buffer()
            self.ser.write(b"GET_MODE\nGET_INFO\nGET_IMU\nGET_CONTROL\nSTREAM,1\n")
            self.status = f"CONNECTED: {port}"
        except (serial.SerialException, OSError) as exc:
            self.close()
            self.status = f"CONNECTION FAILED: {port}"
            print(self.status, exc)

    def accept_line(self, message):
        if message in ("CONTROL,PC", "CONTROL,RULES"):
            mode = message.split(",")[1]
            if mode != self.operating_mode:
                self.rule_state = None
                self.samples.clear()
            self.operating_mode = mode
        elif message.startswith("RULE_STATE,"):
            try:
                fields = message.split(",")
                if len(fields) != 7 or fields[2] not in ("READY", "CALIBRATING"):
                    raise ValueError("invalid rule state")
                stamp, count, previous, age = (int(fields[i]) for i in (1, 3, 4, 5))
                if any(not 0 <= v < 2**32 for v in (stamp, count, previous, age)):
                    raise ValueError("invalid rule counter")
                result = {"NONE": "--", "FIRST": "FIRST BITE", "OK": "OK", "TOO_FAST": "TOO FAST"}[
                    fields[6]
                ]
                if self.operating_mode == "RULES" and self.latest is not None:
                    # Signed delta handles the shared 32-bit micros() rollover.
                    delta = (stamp - self.last_us + 2**31) % 2**32 - 2**31
                    if abs(delta) <= 500000:
                        self.rule_state = RuleState(
                            self.latest.t + delta / 1e6,
                            fields[2] == "READY",
                            count,
                            previous / 1000 if count > 1 else None,
                            age / 1000,
                            result,
                        )
            except (ValueError, IndexError, KeyError):
                self.invalid_frames += 1
        elif message.startswith("MODE,"):
            try:
                value = int(message.split(",")[1])
                if 0 < value <= 3600000:
                    self.target_ms = value
            except (ValueError, IndexError):
                pass
        elif message.startswith("IMU,"):
            try:
                fields = message.split(",")
                if len(fields) != 9:
                    raise ValueError("wrong frame length")
                us = int(fields[1])
                values = np.asarray([float(v) for v in fields[2:]], dtype=float)
                if not 0 <= us < 2**32 or not np.isfinite(values).all():
                    raise ValueError("non-finite frame")
                if np.max(np.abs(values[:3])) > 40 or np.max(np.abs(values[3:6])) > 9:
                    raise ValueError("frame exceeds configured sensor range")
                restart = False
                if self.last_us is not None:
                    if us == self.last_us:
                        return
                    if us < self.last_us:
                        if self.last_us > 0xF0000000 and us < 0x0FFFFFFF:
                            self.time_offset += 2**32
                        else:
                            self.time_offset += self.last_us + 20000 - us
                            restart = True
                            self.rule_state = None
                sample = IMUSample(
                    (us + self.time_offset) / 1e6, values[:3], values[3:6], values[6], restart
                )
                self.last_us = us
                self.latest = sample
                self.samples.append(sample)
                self.last_rx = time.monotonic()
                self.imu_status = f"{self.chip_name} LIVE"
            except (ValueError, OverflowError):
                self.invalid_frames += 1
        elif message.startswith("IMU_STATUS,"):
            self.imu_status = message[len("IMU_STATUS,") :]
            fields = message.split(",")
            if len(fields) >= 4:
                self.chip_name = {"0x68": "MPU6050-compatible", "0x70": "MPU6500-compatible"}.get(
                    fields[3], "UNKNOWN IMU"
                )
            if not message.startswith("IMU_STATUS,OK"):
                self.last_rx = None
                self.samples.clear()
                self.rule_state = None
        elif message.startswith("INFO,"):
            self.info = message

    @property
    def live(self):
        return (
            self.ser is not None
            and self.last_rx is not None
            and time.monotonic() - self.last_rx < 0.5
        )

    @property
    def target_s(self):
        return self.target_ms / 1000.0

    @property
    def mode_name(self):
        return MODE_NAMES.get(self.target_ms, "CUSTOM")

    def poll(self):
        if self.ser is None:
            return
        try:
            count = min(self.ser.in_waiting, 65536)
            if count:
                self.rx_buffer.extend(self.ser.read(count))
            while b"\n" in self.rx_buffer:
                raw, _, remaining = self.rx_buffer.partition(b"\n")
                self.rx_buffer = bytearray(remaining)
                self.accept_line(raw.decode("ascii", errors="replace").strip())
            if len(self.rx_buffer) > 1024:
                self.rx_buffer.clear()
                self.invalid_frames += 1
            self._service_host()
        except (serial.SerialException, OSError) as exc:
            print("Arduino disconnected:", exc)
            self.close()
            self.status = "DISCONNECTED"

    def drain(self):
        samples = list(self.samples)
        self.samples.clear()
        return samples

    def _send(self, command):
        if self.ser is None:
            return False
        try:
            self.ser.write(command.encode("ascii") + b"\n")
            return True
        except (serial.SerialException, OSError):
            self.close()
            self.status = "DISCONNECTED"
            return False

    def _service_host(self):
        if not self.host_enabled or self.operating_mode == "UNKNOWN":
            return
        if self._requested_control is not None:
            mode = self._requested_control
            self._requested_control = None
            self.select_control(mode)
        now = time.monotonic()
        if now - self._last_ping >= 1.0:
            self._last_ping = now
            self._send("HOST_PING")

    def select_control(self, mode):
        if mode not in ("PC", "RULES"):
            raise ValueError("Unknown control mode")
        if self.operating_mode == "UNKNOWN":
            return False  # Old firmware cannot provide standalone detection.
        if mode == "PC" and not self.host_enabled:
            return False
        return self._send(f"SET_CONTROL,{mode}")

    def calibrate_rules(self):
        if self.operating_mode == "RULES":
            self.rule_state = None
            self._send("CALIBRATE")

    def alert(self):
        if self.operating_mode == "RULES":
            return
        print("TOO FAST -> ALERT")
        self._send("ALERT")

    def close(self):
        if self.ser is not None:
            try:
                if self.host_enabled and self.operating_mode == "PC":
                    self.ser.write(b"SET_CONTROL,RULES\n")
            except (serial.SerialException, OSError):
                pass
            try:
                self.ser.close()
            except OSError:
                pass
        self.ser = None
        self.last_rx = None
        self.samples.clear()
        self.rule_state = None


def multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def gravity_in_body(q):
    w, x, y, z = q
    return np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


class RealIMU:
    """VQF 6D attitude with still calibration and gyro bias tracking; yaw is unreferenced."""

    def __init__(self):
        self.segment = 0
        self.reset()

    def reset(self):
        self.segment += 1
        self.window = deque(maxlen=100)
        self.calibrated = False
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.bias = np.zeros(3)
        self.neutral = np.array([0.0, 0.0, 1.0])
        self.last_t = None
        self.tilt = 0.0
        self.corrected_gyro = np.zeros(3)
        self.stationary = False
        self.bias_sigma = 0.0
        self._filter = None
        self._seed_bias = np.zeros(3)
        self._period = 0.02
        self._filter_time = None
        self._previous_sample = None
        self.status = "HOLD SENSOR STILL: 0/100"

    def _finish_update(self, sample):
        residual_bias, self.bias_sigma = self._filter.getBiasEstimate()
        self.bias = self._seed_bias + residual_bias
        self.corrected_gyro = sample.gyro - self.bias
        self.q = self._filter.getQuat6D()
        self.stationary = self._filter.getRestDetected()
        self.status = (
            "CALIBRATED / REST (BIAS TRACKING)" if self.stationary else "CALIBRATED / LIVE"
        )
        self.tilt = math.degrees(
            math.acos(float(np.clip(np.dot(gravity_in_body(self.q), self.neutral), -1, 1)))
        )

    def update(self, sample):
        dt = 0.02 if self.last_t is None else sample.t - self.last_t
        if sample.restart or dt <= 0 or dt > 0.25:
            self.reset()
            dt = 0.02
        self.last_t = sample.t
        if not self.calibrated:
            self.window.append(sample)
            self.status = f"HOLD SENSOR STILL: {len(self.window)}/100"
            if len(self.window) < 100:
                return
            acceleration = np.array([p.accel for p in self.window])
            gyro = np.array([p.gyro for p in self.window])
            mean_accel = acceleration.mean(axis=0)
            # Slow tilt would contaminate the initial gyro bias estimate.
            first, second = acceleration[:50].mean(axis=0), acceleration[50:].mean(axis=0)
            direction_change = math.degrees(
                math.acos(
                    float(
                        np.clip(
                            np.dot(first, second)
                            / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12),
                            -1,
                            1,
                        )
                    )
                )
            )
            if (
                np.max(acceleration.std(axis=0)) > 0.07
                or np.max(gyro.std(axis=0)) > 0.0045
                or direction_change > 0.4
                or abs(np.linalg.norm(mean_accel) - 9.80665) > 1.2
                or np.linalg.norm(gyro.mean(axis=0)) > 0.25
            ):
                self.status = "KEEP STILL FOR 2 SECONDS TO CALIBRATE"
                return
            if VQF is None:
                self.status = "VQF MISSING: py -3.11 -m pip install vqf==2.1.2"
                return
            # Subtract static bias first; VQF tracks residual bias up to 0.5 deg/s.
            self._seed_bias = gyro.mean(axis=0)
            self._period = float(
                np.clip(np.median(np.diff([s.t for s in self.window])), 0.01, 0.04)
            )
            self._filter = VQF(
                self._period,
                tauAcc=2.0,
                restMinT=1.0,
                restThGyr=0.4,
                restThAcc=0.10,
                biasClip=0.5,
                motionBiasEstEnabled=True,
                restBiasEstEnabled=True,
            )
            # Initialize VQF from the calibration window.
            for point in self.window:
                self._filter.update(
                    np.ascontiguousarray(point.gyro - self._seed_bias),
                    np.ascontiguousarray(point.accel),
                )
            self.q = self._filter.getQuat6D()
            self.neutral = gravity_in_body(self.q)
            self._filter_time = sample.t
            self._previous_sample = sample
            self.calibrated = True
            self._finish_update(sample)
            return
        # Resample device timestamps onto VQF's fixed-period clock.
        previous = self._previous_sample
        next_t = self._filter_time + self._period
        while next_t <= sample.t + 1e-9:
            fraction = float(np.clip((next_t - previous.t) / (sample.t - previous.t), 0.0, 1.0))
            gyro = previous.gyro + fraction * (sample.gyro - previous.gyro) - self._seed_bias
            accel = previous.accel + fraction * (sample.accel - previous.accel)
            self._filter.update(np.ascontiguousarray(gyro), np.ascontiguousarray(accel))
            self._filter_time = next_t
            next_t += self._period
        self._previous_sample = sample
        self._finish_update(sample)

    @property
    def roll_pitch(self):
        w, x, y, z = self.q
        return (
            math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
            math.degrees(math.asin(float(np.clip(2 * (w * y - z * x), -1, 1)))),
        )
