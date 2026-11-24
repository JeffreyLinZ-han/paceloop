"""Map forearm IMU orientation to arm targets using a taught lifting direction."""

import math

import numpy as np

from paceloop.bridge.arduino import gravity_in_body, multiply


def conjugate(q):
    return np.asarray(q) * np.array([1.0, -1.0, -1.0, -1.0])


def rotate(q, vector):
    return multiply(multiply(q, np.r_[0.0, vector]), conjugate(q))[1:]


class ForearmFollower:
    driven = ("J1", "J3", "J4", "J5")

    def __init__(self, controls):
        self.controls = controls
        self.indices = {s.name: i for i, s in enumerate(controls)}
        self.reset([s.home for s in controls])

    def reset(self, base):
        self.base = np.array(base, dtype=float)
        self.commands = self.base.copy()
        self.q0 = None
        self.forward = None
        self.up = None
        self.side = None
        self.last_t = None
        self.angles = np.zeros(3)
        self.previous_raw = np.zeros(3)
        self.limited = False
        self.status = "HOLD FOREARM LEVEL AND STILL FOR CALIBRATION"

    def update(self, quaternion, sensor_time):
        q = np.asarray(quaternion, dtype=float)
        if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
            self.status = "INVALID ATTITUDE / HOLDING"
            return self.commands.copy()
        q = q / np.linalg.norm(q)
        if self.q0 is None:
            self.q0 = q.copy()
            self.up = gravity_in_body(q)
            self.last_t = sensor_time
        dt = np.clip(sensor_time - self.last_t, 0.0, 0.1)
        self.last_t = sensor_time
        relative = multiply(conjugate(self.q0), q)
        if relative[0] < 0:
            relative = -relative
        if self.forward is None:
            sine = np.linalg.norm(relative[1:])
            vector = (
                relative[1:] * (2 * math.atan2(sine, relative[0]) / sine)
                if sine > 1e-8
                else np.zeros(3)
            )
            lift_axis = vector - np.dot(vector, self.up) * self.up
            angle = np.linalg.norm(lift_axis)
            if angle < math.radians(12):
                self.status = "RAISE FOREARM ONCE (12+ DEG) TO TEACH LIFT DIRECTION"
                return self.commands.copy()
            # Neutral forearm is horizontal; positive lift rotates forward toward up.
            self.forward = np.cross(self.up, lift_axis / angle)
            self.forward /= np.linalg.norm(self.forward)
            self.side = np.cross(self.up, self.forward)

        direction = rotate(relative, self.forward)
        forward = float(np.dot(direction, self.forward))
        side = float(np.dot(direction, self.side))
        vertical = float(np.dot(direction, self.up))
        horizontal = math.hypot(forward, side)
        lift = math.degrees(math.atan2(vertical, horizontal))
        # Heading is indeterminate for a vertical forearm: retain it there.
        heading = math.degrees(math.atan2(side, forward)) if horizontal > 0.2 else self.angles[1]
        swing = np.r_[1 + np.dot(self.forward, direction), np.cross(self.forward, direction)]
        if np.linalg.norm(swing) < 1e-6:
            twist = self.angles[2]
        else:
            swing /= np.linalg.norm(swing)
            twist_q = multiply(conjugate(swing), relative)
            twist = math.degrees(2 * math.atan2(np.dot(twist_q[1:], self.forward), twist_q[0]))
            twist = (twist + 180) % 360 - 180
        raw = np.array([lift, heading, twist])
        raw[1:] += 360 * np.round((self.previous_raw[1:] - raw[1:]) / 360)
        self.previous_raw = raw.copy()
        bounded = np.clip(raw, [-70, -60, -75], [80, 60, 75])
        self.limited = bool(np.any(np.abs(raw - bounded) > 0.01))
        bounded = np.sign(bounded) * np.maximum(np.abs(bounded) - 0.4, 0.0)
        self.angles += (1 - math.exp(-float(dt) / 0.035)) * (bounded - self.angles)
        lift, heading, twist = self.angles
        self.commands = self.base.copy()
        offsets = {"J1": max(0.0, lift - 40.0), "J3": -heading, "J4": min(lift, 40.0), "J5": twist}
        for name, offset in offsets.items():
            index = self.indices[name]
            spec = self.controls[index]
            desired = self.base[index] + offset
            self.commands[index] = np.clip(desired, spec.lower, spec.upper)
            self.limited |= abs(desired - self.commands[index]) > 0.01
        self.status = "SYNC LIVE" + (" / RANGE LIMITED" if self.limited else "")
        return self.commands.copy()

    def recenter(self, quaternion, sensor_time):
        """Reset heading at a user-confirmed neutral pose, retaining the original anchor."""
        q = np.asarray(quaternion, dtype=float)
        if self.forward is None or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
            return False
        new_up = gravity_in_body(q / np.linalg.norm(q))
        forward = self.forward - np.dot(self.forward, new_up) * new_up
        if np.linalg.norm(forward) < 0.2:
            self.status = "RETURN TO THE ORIGINAL LEVEL POSE BEFORE RECENTERING"
            return False
        self.q0 = q / np.linalg.norm(q)
        self.up = new_up
        self.forward = forward / np.linalg.norm(forward)
        self.side = np.cross(self.up, self.forward)
        self.last_t = sensor_time
        self.angles[:] = 0.0
        self.previous_raw[:] = 0.0
        self.commands = self.base.copy()
        self.limited = False
        self.status = "SYNC LIVE / RECENTERED"
        return True


class LiveControls:
    """Track moving joint targets with continuous position and bounded speed and acceleration."""

    speed = math.radians(150)
    acceleration = math.radians(900)
    bandwidth = 20.0

    def __init__(self, model, data, controls, previous=None):
        self.model, self.data, self.controls = model, data, controls
        self.ids = [model.actuator(s.actuator).id for s in controls]
        self.position = np.array([float(data.ctrl[i]) for i in self.ids])
        self.velocity = np.zeros(len(controls)) if previous is None else previous.velocity.copy()
        self.accel = np.zeros(len(controls))
        self.targets = self.position.copy()
        self.limits = model.actuator_ctrlrange[self.ids].copy()
        self.arm = np.array([not s.gripper for s in controls])

    def set_targets(self, values):
        targets = np.array([s.command(v) for s, v in zip(self.controls, values)])
        if np.isfinite(targets).all():
            self.targets = np.clip(targets, self.limits[:, 0], self.limits[:, 1])

    def step(self, dt):
        error = self.targets - self.position
        # Braking speed follows the remaining distance to each target.
        desired = np.sign(error) * np.minimum.reduce(
            [
                np.full_like(error, self.speed),
                self.bandwidth * np.abs(error),
                np.sqrt(2 * self.acceleration * np.abs(error)),
            ]
        )
        dv = np.clip(desired - self.velocity, -self.acceleration * dt, self.acceleration * dt)
        self.accel = dv / dt
        self.velocity += dv
        position = self.position + self.velocity * dt
        clipped = np.clip(position, self.limits[:, 0], self.limits[:, 1])
        self.velocity[clipped != position] = 0.0
        self.position = clipped
        self.position[~self.arm] = self.targets[~self.arm]
        self.velocity[~self.arm] = 0.0
        self.data.ctrl[self.ids] = self.position
