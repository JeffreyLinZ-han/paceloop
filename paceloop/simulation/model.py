"""Shared model loading and control mapping for the PaceLoop OpenArm twin."""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from paceloop.paths import ROBOT_XML


@dataclass(frozen=True)
class Control:
    name: str
    actuator: str
    lower: float
    upper: float
    home: float
    gripper: bool = False

    def command(self, value):
        return 0.1 + 1.4 * value / 100.0 if self.gripper else math.radians(value)


def compile_model(root, directory):
    """Load through MuJoCo's in-memory VFS, including meshes in Chinese paths."""
    root = ET.fromstring(ET.tostring(root, encoding="unicode"))
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    assets, names = {}, {}
    for asset in root.findall("asset/*"):
        if asset.get("file"):
            source = (Path(directory) / asset.get("file")).resolve()
            if source not in names:
                virtual_name = f"asset_{len(names):04d}{source.suffix}"
                names[source] = virtual_name
                assets[virtual_name] = source.read_bytes()
            asset.set("file", names[source])
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets=assets)


def load_model(path=None):
    path = Path(path) if path else ROBOT_XML
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    model = compile_model(root, path.parent)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    for name in ("imu_accel", "imu_gyro", "imu_quat", "wrist_pos"):
        data.sensor(name)
    return model, data


def controls_for(model):
    controls = []
    for j in range(1, 8):
        name = f"right_joint{j}_ctrl"
        lower, upper = model.actuator(name).ctrlrange
        controls.append(
            Control(
                f"J{j}", name, math.degrees(lower), math.degrees(upper), 90.0 if j == 4 else 0.0
            )
        )
    controls.append(Control("Gripper", "right_bb_gripper_open_close_ctrl", 0, 100, 0, True))
    return controls


def set_controls(data, controls, values):
    for control, value in zip(controls, values):
        data.actuator(control.actuator).ctrl[0] = control.command(value)


class SmoothControls:
    """Continuous position/velocity/acceleration targets, including during reversal.

    Each arm joint follows a quintic trajectory. Duration is chosen from exact
    polynomial extrema to respect 90 deg/s, 360 deg/s^2 and 2400 deg/s^3 limits.
    The original gripper target filter remains active.
    """

    speed = math.radians(90)
    acceleration = math.radians(360)
    jerk = math.radians(2400)

    def __init__(self, model, data, controls):
        self.model, self.data, self.controls = model, data, controls
        self.ids = [model.actuator(spec.actuator).id for spec in controls]
        self.position = np.array([float(data.ctrl[i]) for i in self.ids])
        self.velocity = np.zeros(len(controls))
        self.accel = np.zeros(len(controls))
        self.targets = self.position.copy()
        self.elapsed = np.zeros(len(controls))
        self.duration = np.zeros(len(controls))
        self.coefficients = np.zeros((len(controls), 6))
        self.coefficients[:, 0] = self.position

    @staticmethod
    def _extrema(poly):
        derivative = np.polynomial.polynomial.polyder(poly)
        roots = np.polynomial.polynomial.polyroots(derivative)
        points = [0.0, 1.0] + [r.real for r in roots if abs(r.imag) < 1e-8 and 0 < r.real < 1]
        return np.polynomial.polynomial.polyval(points, poly)

    def _plan(self, i, target):
        p, v, a = self.position[i], self.velocity[i], self.accel[i]
        distance = abs(target - p)
        duration = max(
            0.25,
            1.9 * distance / self.speed,
            math.sqrt(6 * distance / self.acceleration),
            (60 * distance / self.jerk) ** (1 / 3),
        )
        limits = self.model.actuator_ctrlrange[self.ids[i]]
        for _ in range(80):
            # Normalized time u=t/T gives a well-conditioned polynomial.
            c = np.zeros(6)
            c[:3] = p, v * duration, 0.5 * a * duration**2
            displacement = target - p
            c[3] = 10 * displacement - 6 * v * duration - 1.5 * a * duration**2
            c[4] = -15 * displacement + 8 * v * duration + 1.5 * a * duration**2
            c[5] = 6 * displacement - 3 * v * duration - 0.5 * a * duration**2
            derivatives = [
                np.polynomial.polynomial.polyder(c, order) / duration**order for order in (1, 2, 3)
            ]
            bounded = all(
                np.max(np.abs(self._extrema(poly))) <= limit * 1.000001
                for poly, limit in zip(derivatives, (self.speed, self.acceleration, self.jerk))
            )
            positions = self._extrema(c)
            within_joints = (
                min(positions) >= limits[0] - 1e-9 and max(positions) <= limits[1] + 1e-9
            )
            if bounded and within_joints:
                self.coefficients[i], self.duration[i], self.elapsed[i] = c, duration, 0.0
                self.targets[i] = target
                return True
            duration *= 1.08
        # Near a joint limit, finish the previously feasible stop before retargeting.
        return False

    def set_targets(self, values):
        for i, (spec, value) in enumerate(zip(self.controls, values)):
            limits = self.model.actuator_ctrlrange[self.ids[i]]
            target = float(np.clip(spec.command(value), *limits))
            if spec.gripper:
                self.targets[i] = target
            elif abs(target - self.targets[i]) > 1e-9:
                self._plan(i, target)

    def step(self, dt):
        for i, spec in enumerate(self.controls):
            if spec.gripper:
                self.data.ctrl[self.ids[i]] = self.targets[i]
                continue
            self.elapsed[i] = min(self.duration[i], self.elapsed[i] + dt)
            if self.elapsed[i] >= self.duration[i]:
                self.position[i] = self.targets[i]
                self.velocity[i] = self.accel[i] = 0.0
            else:
                u = self.elapsed[i] / self.duration[i]
                c = self.coefficients[i]
                self.position[i] = np.polynomial.polynomial.polyval(u, c)
                self.velocity[i] = (
                    np.polynomial.polynomial.polyval(u, np.polynomial.polynomial.polyder(c))
                    / self.duration[i]
                )
                self.accel[i] = (
                    np.polynomial.polynomial.polyval(u, np.polynomial.polynomial.polyder(c, 2))
                    / self.duration[i] ** 2
                )
            self.data.ctrl[self.ids[i]] = self.position[i]


def bite_angle(data):
    """Upward elbow excursion from the 90-degree home pose; simulation proxy only."""
    return max(0.0, math.degrees(float(data.joint("openarm_right_joint4").qpos[0])) - 90.0)
