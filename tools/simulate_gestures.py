"""Generate labeled OpenArm wrist IMU episodes with prescribed joint motion.

MuJoCo inverse dynamics evaluates the sensor response at the prescribed position,
velocity and acceleration. No robot coordinates are supplied to the classifier.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gesture_model import LABELS, SAMPLE_HZ
from openarm_model import load_model

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_VERSION = 1


def smoothstep(u):
    u = np.clip(u, 0, 1)
    return u**3 * (10 - 15 * u + 6 * u * u)


class GestureSimulator:
    def __init__(self):
        self.model, self.data = load_model(ROOT / "arm.xml")
        # Prescribed trajectories do not model skin, food or contact forces.
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.qids = np.array(
            [self.model.joint(f"openarm_right_joint{j}").qposadr[0] for j in range(1, 8)]
        )
        self.vids = np.array(
            [self.model.joint(f"openarm_right_joint{j}").dofadr[0] for j in range(1, 8)]
        )
        self.ranges = self.model.jnt_range[
            [self.model.joint(f"openarm_right_joint{j}").id for j in range(1, 8)]
        ]

    def episode(self, session, repetition, label, stress=False):
        # All classes share the same distribution of mount, offsets and noise.
        wearer = np.random.default_rng(session)
        rng = np.random.default_rng(np.random.SeedSequence([session, repetition, label, 9173]))
        axis = wearer.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(wearer.uniform(-45 if stress else -25, 45 if stress else 25))
        quat = np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quat)
        mount = matrix.reshape(3, 3)
        accel_bias = wearer.normal(0, 0.12, 3)
        gyro_bias = wearer.normal(0, 0.035, 3)
        base = np.array([0, 10, 0, 85, 0, 0, 0], dtype=float) + rng.normal(
            0, [7, 5, 8, 8, 12, 4, 5]
        )
        variants = {
            1: ([25, 12, -25, 128, 10, -8, 10], [10, 6, 10, 7, 18, 6, 8]),
            2: ([48, 58, 15, 125, 25, 0, 10], [12, 15, 15, 8, 20, 9, 12]),
            3: ([33, 22, -3, 130, -10, 12, 0], [10, 10, 13, 6, 20, 8, 10]),
        }
        if label:
            center, spread = variants[label]
            target = np.array(center) + rng.normal(0, spread)
        else:
            # Rest, reach, lateral gesture and a deliberately high non-bite lift.
            choices = [
                base,
                [10, 15, 5, 55, 10, 0, 0],
                [35, 65, 35, 95, -35, 5, 0],
                [5, 8, 15, 125, -35, 0, 0],
            ]
            target = np.array(choices[(session + repetition) % len(choices)]) + rng.normal(
                0, [10, 8, 10, 8, 12, 5, 5]
            )
        speed = rng.uniform(0.65, 1.5) if stress else rng.uniform(0.8, 1.25)
        pre = rng.uniform(2.5, 3.0)
        reach = rng.uniform(0.8, 1.6) * speed
        hold = rng.uniform(0.7, 1.6)
        retreat = rng.uniform(0.8, 1.7) * speed
        end_hold = pre + reach + hold
        t = np.arange(0, end_hold + retreat + rng.uniform(1.0, 1.6), 1 / SAMPLE_HZ)
        progress = smoothstep((t - pre) / reach) - smoothstep((t - end_hold) / retreat)
        q = base + progress[:, None] * (target - base)
        if label in (2, 3):
            # Scratching includes repeated wrist motion; face touching can be a
            # single rub or a still hold, so not all negatives get oscillations.
            duration = hold
            u = np.clip((t - pre - reach) / duration, 0, 1)
            envelope = np.sin(np.pi * u) ** 2
            frequency = rng.uniform(1.5, 3.5) if label == 2 else rng.uniform(0.3, 1.2)
            amplitude = rng.uniform(4, 13) if label == 2 else rng.uniform(0, 6)
            q[:, 5] += envelope * amplitude * np.sin(2 * np.pi * frequency * (t - pre - reach))
            q[:, 3] += (
                envelope * amplitude * 0.2 * np.cos(2 * np.pi * frequency * (t - pre - reach))
            )
        q = np.clip(np.deg2rad(q), self.ranges[:, 0] + 0.005, self.ranges[:, 1] - 0.005)
        velocity = np.gradient(q, 1 / SAMPLE_HZ, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, 1 / SAMPLE_HZ, axis=0, edge_order=2)
        signal = np.empty((len(t), 6), dtype=np.float32)
        positions = np.empty((len(t), 3), dtype=np.float32)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        for i in range(len(t)):
            self.data.qpos[self.qids] = q[i]
            self.data.qvel[:] = 0
            self.data.qacc[:] = 0
            self.data.qvel[self.vids] = velocity[i]
            self.data.qacc[self.vids] = acceleration[i]
            self.data.time = t[i]
            mujoco.mj_inverse(self.model, self.data)
            signal[i, :3] = self.data.sensor("imu_accel").data @ mount
            signal[i, 3:] = self.data.sensor("imu_gyro").data @ mount
            positions[i] = self.data.sensor("wrist_pos").data
        signal[:, :3] += accel_bias + rng.normal(0, rng.uniform(0.025, 0.12), (len(t), 3))
        signal[:, 3:] += gyro_bias + rng.normal(0, rng.uniform(0.003, 0.018), (len(t), 3))
        signal[:, :3] = np.clip(signal[:, :3], -39.2266, 39.2266)
        signal[:, 3:] = np.clip(signal[:, 3:], -8.72665, 8.72665)
        # The classifier sees the approach/hold portion; rest and return are other.
        active = (t >= pre + reach * 0.65) & (t <= end_hold + retreat * 0.15)
        labels = np.where(active, label, 0).astype(np.int64)
        if not np.isfinite(signal).all():
            raise ValueError("Non-finite simulated IMU")
        return dict(
            time=t,
            signal=signal,
            labels=labels,
            joints=q,
            wrist=positions,
            session=session,
            repetition=repetition,
            action=label,
            event_interval=[float(t[active][0]), float(t[active][-1])] if label else [],
        )


def generate(output, sessions=92, repetitions=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    simulator = GestureSimulator()
    manifest = dict(
        generator_version=GENERATOR_VERSION,
        sample_hz=SAMPLE_HZ,
        labels=list(LABELS),
        source="OpenArm prescribed-motion simulation",
        real_world_validated=False,
        episodes=[],
    )
    for n in range(sessions):
        split = "train" if n < sessions * 0.65 else "validation" if n < sessions * 0.8 else "test"
        for repetition in range(repetitions):
            for label in range(len(LABELS)):
                episode = simulator.episode(12000 + n, repetition, label)
                name = f"s{n:03d}_r{repetition}_{LABELS[label]}.npz"
                np.savez_compressed(output / name, **episode)
                manifest["episodes"].append(
                    dict(file=name, split=split, session=episode["session"], action=label)
                )
        if (n + 1) % 10 == 0:
            print(f"Simulated {n + 1}/{sessions} sessions", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "gestures_v1")
    parser.add_argument("--sessions", type=int, default=92)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    if args.sessions < 20 or args.repetitions < 1:
        parser.error("Use at least 20 sessions and one repetition")
    manifest = generate(args.output, args.sessions, args.repetitions)
    print(f"Saved {len(manifest['episodes'])} labeled episodes to {args.output}")


if __name__ == "__main__":
    main()
