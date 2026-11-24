"""OpenArm viewer, forearm IMU tracking, and gesture pacing."""

import argparse
import csv
import math
import time
import tkinter as tk
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer

from paceloop.bridge.arduino import VQF, ArduinoBridge, RealIMU
from paceloop.bridge.follow import ForearmFollower, LiveControls
from paceloop.learning.model import DEFAULT_MODEL_FILE, GestureRecognizer, TinyGestureModel
from paceloop.paths import ROBOT_XML, ROOT
from paceloop.simulation.model import SmoothControls, bite_angle, controls_for, load_model

# Legacy tilt proxy thresholds: simulated J4 excursion or calibrated IMU tilt.
BITE_ANGLE_DEG = 25.0
RESET_ANGLE_DEG = 10.0
GYRO_THRESHOLD_RAD_S = 0.15
MIN_BITE_GAP_S = 1.2

MODE_NAMES = {3000: "FAST", 5000: "MEDIUM", 15000: "SLOW"}


class BiteTimer:
    def __init__(self):
        self.armed = True
        self.last_bite_time = None
        self.previous_interval = None
        self.bite_count = 0
        self.last_result = "--"

    def step(self, sim_time, lift_deg, gyro_xyz, target_s):
        """Return 'FIRST', 'OK', 'TOO FAST', or None on each physics step."""
        angular_speed = math.sqrt(sum(float(g) ** 2 for g in gyro_xyz))

        # A full return to a lower angle is required before another bite.
        if lift_deg < RESET_ANGLE_DEG:
            self.armed = True

        if not (self.armed and lift_deg > BITE_ANGLE_DEG and angular_speed > GYRO_THRESHOLD_RAD_S):
            return None

        return self.record_event(sim_time, target_s)

    def record_event(self, sim_time, target_s):
        """Record an event from either the tilt proxy or the learned classifier."""
        if self.last_bite_time is not None:
            interval = sim_time - self.last_bite_time
            if interval < MIN_BITE_GAP_S:
                return None
            self.previous_interval = interval
            result = "TOO FAST" if interval < target_s else "OK"
        else:
            result = "FIRST BITE"

        self.armed = False
        self.last_bite_time = sim_time
        self.bite_count += 1
        self.last_result = result
        print(
            f"Bite #{self.bite_count} @ {sim_time:.2f}s | "
            f"previous={self.previous_interval} | "
            f"target={target_s:.1f}s | {result}"
        )
        return result

    def current_interval(self, sim_time):
        if self.last_bite_time is None:
            return None
        return max(0.0, sim_time - self.last_bite_time)


class PaceLoopApp:
    def __init__(
        self,
        model,
        data,
        bridge,
        csv_path=None,
        imu_source="sim",
        follow_imu=True,
        gesture_model=None,
    ):
        self.model = model
        self.data = data
        self.bridge = bridge
        self.hardware_imu = imu_source == "hardware"
        self.real_imu = RealIMU()
        self.gesture = GestureRecognizer(gesture_model) if gesture_model is not None else None
        self._operating_mode = bridge.operating_mode
        self.was_live = False
        self.last_flush = time.monotonic()
        self.recenter_count = 0
        self.timer = BiteTimer()
        self.running = True
        self.control_specs = controls_for(model)
        self.follow_enabled = self.hardware_imu and follow_imu
        self.follower = ForearmFollower(self.control_specs)
        controller = LiveControls if self.follow_enabled else SmoothControls
        self.motion = controller(model, data, self.control_specs)
        self.accumulator = 0.0
        self.history = [deque(maxlen=200) for _ in range(6)]
        self.colors = ["#ff6b6b", "#ffd166", "#06d6a0", "#4dabf7", "#c77dff", "#f4a261"]

        self.root = tk.Tk()
        source = "REAL IMU" if self.hardware_imu else "VIRTUAL IMU"
        self.root.title(f"PaceLoop | Full OpenArm | {source}")
        self.root.geometry("1020x860")
        self.root.minsize(900, 760)

        self._make_controls()
        self._make_timing_panel()
        self._make_sensor_panel()

        if csv_path is None:
            csv_path = ROOT / "logs" / time.strftime("openarm_%Y%m%d_%H%M%S.csv")
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = self.csv_path.open("x", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(
            [
                "sim_time_s",
                *[f"j{j}_cmd_deg" for j in range(1, 8)],
                "gripper_close_pct",
                "ax",
                "ay",
                "az",
                "gx",
                "gy",
                "gz",
                "quat_w",
                "quat_x",
                "quat_y",
                "quat_z",
                "x",
                "y",
                "z",
                "bite_event",
                "current_interval_s",
                "previous_interval_s",
                "mode",
                "target_interval_s",
                "last_result",
                "imu_source",
                "sensor_time_s",
                "tilt_deg",
                "imu_status",
                "temperature_C",
                "sensor_model",
                "sync_state",
                "forearm_lift_deg",
                "forearm_heading_deg",
                "forearm_twist_deg",
                *[f"j{j}_actual_deg" for j in range(1, 8)],
                "gyro_bias_x",
                "gyro_bias_y",
                "gyro_bias_z",
                "imu_at_rest",
                "recenter_count",
                "gesture_label",
                "bite_model_score",
                "gesture_model_source",
                "control_mode",
            ]
        )

        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        )
        with self.viewer.lock():
            self.viewer.cam.lookat[:] = (0.12, 0.0, 0.50)
            self.viewer.cam.distance = 1.65
            self.viewer.cam.azimuth = 150
            self.viewer.cam.elevation = -15
            self.viewer.opt.geomgroup[3] = 0  # CAD collision hulls are not the visible surfaces.
        self.last_wall_time = time.perf_counter()
        self.wall_start = self.last_wall_time
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.tick)

    def _make_controls(self):
        label = (
            "GESTURE TIMING | PC model / Arduino rules"
            if self.hardware_imu
            else (
                "FULL OPENARM | right arm follows forearm IMU | left arm lowered"
                if self.hardware_imu
                else "FULL OPENARM | smooth right-arm motion | left arm lowered"
            )
        )
        frame = tk.LabelFrame(self.root, text=label, padx=8, pady=4)
        frame.pack(fill="x", padx=12, pady=(8, 4))
        self.controls = {}
        for i, spec in enumerate(self.control_specs):
            line = tk.Frame(frame)
            line.grid(row=i // 2, column=i % 2, sticky="ew", padx=4)
            frame.columnconfigure(i % 2, weight=1)
            label = "Gripper %" if spec.gripper else spec.name + " (deg)"
            tk.Label(line, text=label, width=10, anchor="w").pack(side="left")
            slider = tk.Scale(
                line,
                from_=math.ceil(spec.lower - 0.001),
                to=math.floor(spec.upper + 0.001),
                resolution=1,
                orient="horizontal",
                length=320,
                showvalue=True,
            )
            slider.set(spec.home)
            slider.pack(side="left", fill="x", expand=True)
            self.controls[spec.name] = slider
        actions = tk.Frame(frame)
        actions.grid(row=4, column=0, columnspan=2, sticky="w", pady=4)
        self.home_button = tk.Button(actions, text="Return / home", command=self.home_targets)
        self.home_button.pack(side="left", padx=4)
        lift_label = "Lift preview (J4)" if self.hardware_imu else "Lift / bite test (J4)"
        self.lift_button = tk.Button(
            actions, text=lift_label, command=lambda: self.controls["J4"].set(130)
        )
        self.lift_button.pack(side="left", padx=4)
        tk.Label(actions, text="Gripper: 0% open / 100% closed; return between bites").pack(
            side="left", padx=8
        )
        if self.hardware_imu:
            sync = tk.Frame(frame)
            sync.grid(row=5, column=0, columnspan=2, sticky="ew", pady=4)
            self.sync_var = tk.BooleanVar(value=self.follow_enabled)
            tk.Checkbutton(
                sync, text="Follow forearm IMU", variable=self.sync_var, command=self.toggle_sync
            ).pack(side="left", padx=4)
            self.sync_label = tk.Label(
                sync, text="Hold forearm level and still, then lift once", anchor="w", fg="#986000"
            )
            self.sync_label.pack(side="left", padx=8)
            self._set_control_states()

    def _set_control_states(self):
        state = "disabled" if self.follow_enabled else "normal"
        for spec in self.control_specs:
            if not spec.gripper:
                self.controls[spec.name].config(state=state)
        self.home_button.config(state=state)
        self.lift_button.config(state=state)

    def _motion_commands(self):
        return [
            (self.motion.targets[i] - 0.1) / 1.4 * 100
            if spec.gripper
            else math.degrees(self.motion.position[i])
            for i, spec in enumerate(self.control_specs)
        ]

    def _reset_follow(self):
        if self.follow_enabled:
            held = self._motion_commands()
            self.follower.reset(held)
            self.motion.set_targets(held)

    def toggle_sync(self):
        self.follow_enabled = self.sync_var.get()
        held = self._motion_commands()
        with self.viewer.lock():
            if self.follow_enabled:
                self.motion = LiveControls(self.model, self.data, self.control_specs, self.motion)
            else:
                self.motion = SmoothControls(self.model, self.data, self.control_specs)
            self.motion.set_targets(held)
        self.follower.reset(held)
        self._set_control_states()
        for spec, value in zip(self.control_specs, held):
            self.controls[spec.name].set(value)

    def home_targets(self):
        for spec in self.control_specs:
            self.controls[spec.name].set(spec.home)

    def _make_timing_panel(self):
        label = (
            "GESTURE TIMING | simulation-trained IMU classifier"
            if self.gesture is not None
            else "GESTURE TIMING | experimental tilt-based bite proxy"
            if self.hardware_imu
            else "BITE TIMING"
        )
        frame = tk.LabelFrame(self.root, text=label, padx=8, pady=8)
        frame.pack(fill="x", padx=12, pady=4)
        for col in range(4):
            frame.columnconfigure(col, weight=1)

        self.metrics = {}
        items = (
            ("Interval Status", "WAITING"),
            ("Current Interval", "--"),
            ("Previous Interval", "--"),
            ("Pace / Target", "MEDIUM / 5.0 s"),
        )
        for col, (title, initial) in enumerate(items):
            cell = tk.Frame(frame, padx=6)
            cell.grid(row=0, column=col, sticky="ew")
            tk.Label(cell, text=title, font=("Segoe UI", 10), anchor="w").pack(anchor="w")
            value = tk.Label(
                cell, text=initial, font=("Consolas", 14, "bold"), fg="#0b6e75", anchor="w"
            )
            value.pack(anchor="w", pady=(4, 2))
            self.metrics[title] = value

        self.details = tk.Label(
            frame, text="Bites: 0    |    Last Result: --", font=("Consolas", 11), anchor="w"
        )
        self.details.grid(row=1, column=0, columnspan=4, sticky="w", padx=6, pady=(8, 0))
        if self.gesture is not None or self.hardware_imu:
            self.gesture_status = tk.Label(frame, text="Gesture model: warming up", anchor="w")
            self.gesture_status.grid(row=2, column=0, columnspan=4, sticky="w", padx=6)
        if self.bridge.ser is not None or self.hardware_imu:
            self.control_status = tk.Label(frame, text="Checking Arduino mode...", anchor="w")
            self.control_status.grid(row=3, column=0, columnspan=3, sticky="w", padx=6)
            self.mode_button = tk.Button(frame, text="Switch mode", command=self.switch_mode)
            self.mode_button.grid(row=3, column=3, sticky="e", padx=6)

    def _make_sensor_panel(self):
        label = "REAL IMU | Arduino USB / I2C" if self.hardware_imu else "VIRTUAL IMU (MuJoCo)"
        frame = tk.LabelFrame(self.root, text=label, padx=8, pady=8)
        frame.pack(fill="both", expand=True, padx=12, pady=4)
        if self.hardware_imu:
            actions = tk.Frame(frame)
            actions.pack(fill="x", pady=(0, 4))
            tk.Button(
                actions,
                text="Calibrate neutral: hold still for 2 seconds",
                command=self.calibrate_imu,
            ).pack(side="left")
            tk.Button(actions, text="Recenter to start [F9]", command=self.recenter_imu).pack(
                side="left", padx=8
            )
            tk.Label(actions, text="Return to your starting pose first").pack(side="left")
            self.root.bind("<F9>", lambda event: self.recenter_imu())
        self.imu_label = tk.Label(frame, font=("Consolas", 10), justify="left", anchor="w")
        self.imu_label.pack(fill="x")
        self.canvas = tk.Canvas(frame, bg="#101820", height=250, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=(8, 2))
        self.footer = tk.Label(self.root, text="", font=("Segoe UI", 9), anchor="w")
        self.footer.pack(fill="x", padx=16, pady=(0, 8))

    def calibrate_imu(self, reset_board=True):
        if reset_board:
            self.bridge.calibrate_rules()
        self.real_imu.reset()
        self.timer = BiteTimer()
        if getattr(self, "gesture", None) is not None:
            self.gesture.reset()
        self._reset_follow()
        for values in self.history:
            values.clear()

    @property
    def rules_mode(self):
        return getattr(self.bridge, "operating_mode", "UNKNOWN") == "RULES"

    def switch_mode(self):
        self.bridge.select_control("PC" if self.rules_mode else "RULES")

    def _sync_operating_mode(self):
        mode = getattr(self.bridge, "operating_mode", "UNKNOWN")
        if mode != getattr(self, "_operating_mode", "UNKNOWN"):
            self._operating_mode = mode
            self.timer = BiteTimer()
            if self.gesture is not None:
                self.gesture.reset()

    def _apply_rule_state(self):
        state = self.bridge.rule_state
        if state is None:
            return None
        new_event = state.count > self.timer.bite_count
        self.timer.bite_count = state.count
        self.timer.previous_interval = state.previous
        self.timer.last_bite_time = state.t - state.age if state.count else None
        self.timer.last_result = state.result
        return state.result if new_event else None

    def recenter_imu(self):
        if not (self.follow_enabled and self.bridge.live and self.real_imu.calibrated):
            return
        if self.follower.recenter(self.real_imu.q, self.bridge.latest.t):
            self.recenter_count += 1
            self.follower.commands[-1] = float(self.controls["Gripper"].get())

    def _log_sample(
        self,
        sim_time,
        commands,
        acc,
        gyro,
        quat,
        pos,
        event,
        sensor_time,
        source,
        tilt="",
        status="",
        temperature="",
    ):
        current = self.timer.current_interval(sensor_time)
        previous = self.timer.previous_interval
        self.writer.writerow(
            [
                f"{sim_time:.3f}",
                *commands,
                *acc,
                *gyro,
                *quat,
                *pos,
                event or "",
                "" if current is None else f"{current:.3f}",
                "" if previous is None else f"{previous:.3f}",
                self.bridge.mode_name,
                f"{self.bridge.target_s:.3f}",
                self.timer.last_result,
                source,
                f"{sensor_time:.6f}",
                tilt,
                status,
                temperature,
                self.bridge.chip_name if source == "hardware" else "MuJoCo",
                self.follower.status if self.follow_enabled else "MANUAL",
                *(self.follower.angles if self.follow_enabled else ["", "", ""]),
                *[
                    math.degrees(float(self.data.joint(f"openarm_right_joint{j}").qpos[0]))
                    for j in range(1, 8)
                ],
                *(self.real_imu.bias if source == "hardware" else ["", "", ""]),
                self.real_imu.stationary if source == "hardware" else "",
                self.recenter_count,
                self.gesture.label if self.gesture is not None and not self.rules_mode else "",
                float(self.gesture.scores[1])
                if self.gesture is not None
                and not self.rules_mode
                and self.gesture.scores is not None
                else "",
                self.gesture.model.metadata["source"]
                if self.gesture is not None and not self.rules_mode
                else "",
                self.bridge.operating_mode if self.bridge.ser is not None else "SIMULATION",
            ]
        )

    def _process_hardware(self, sim_time, commands):
        self._sync_operating_mode()
        live = self.bridge.live
        if self.was_live and not live:
            self.calibrate_imu(reset_board=False)
        self.was_live = live
        samples = self.bridge.drain()
        if not live:
            return
        for sample in samples:
            old_segment = self.real_imu.segment
            self.real_imu.update(sample)
            if self.real_imu.segment != old_segment:
                if not self.rules_mode:
                    self.timer = BiteTimer()
                if self.gesture is not None:
                    self.gesture.reset()
                self._reset_follow()
                for values in self.history:
                    values.clear()
            event = None
            calibrated = self.real_imu.calibrated
            if calibrated:
                if self.follow_enabled:
                    tracked = self.follower.update(self.real_imu.q, sample.t)
                    tracked[-1] = commands[-1]  # Gripper stays under manual control.
                    commands[:] = tracked.tolist()
                if not self.rules_mode and self.gesture is not None:
                    if self.gesture.update(sample.t, sample.accel, sample.gyro, sample.restart):
                        event = self.timer.record_event(sample.t, self.bridge.target_s)
                elif not self.rules_mode:
                    event = self.timer.step(
                        sample.t,
                        self.real_imu.tilt,
                        self.real_imu.corrected_gyro,
                        self.bridge.target_s,
                    )
            if self.rules_mode and sample is samples[-1]:
                event = self._apply_rule_state()
            if event == "TOO FAST" and not self.rules_mode:
                self.bridge.alert()
            for values, value in zip(self.history, [*sample.accel, *sample.gyro]):
                values.append(float(value))
            self._log_sample(
                sim_time,
                commands,
                sample.accel,
                sample.gyro,
                self.real_imu.q if calibrated else [""] * 4,
                [""] * 3,
                event,
                sample.t,
                "hardware",
                self.real_imu.tilt if calibrated else "",
                self.real_imu.status,
                sample.temperature,
            )
        if self.rules_mode and not samples:
            self._apply_rule_state()

    def _draw_graph(self):
        c = self.canvas
        c.delete("all")
        width = max(c.winfo_width(), 200)
        height = max(c.winfo_height(), 200)
        half = height / 2
        for group in range(2):
            top = group * half
            center = top + half / 2
            c.create_line(0, center, width, center, fill="#425466")
            c.create_text(
                8,
                top + 13,
                anchor="w",
                fill="white",
                text="ACCEL (m/s^2)" if group == 0 else "GYRO (rad/s)",
            )
            for j in range(3):
                index = 3 * group + j
                values = list(self.history[index])
                if len(values) < 2:
                    continue
                scale = max(12.0 if group == 0 else 2.0, max(abs(v) for v in values) * 1.3)
                points = []
                for i, val in enumerate(values):
                    points.extend(
                        (10 + i * (width - 20) / 199, center - val * (half * 0.38) / scale)
                    )
                c.create_line(*points, fill=self.colors[index], width=2)
                c.create_text(
                    width - 12,
                    top + 17 + j * 17,
                    anchor="e",
                    fill=self.colors[index],
                    text=f"{'xyz'[j]} {values[-1]:+.2f}",
                )

    def tick(self):
        if not self.running:
            return
        if not self.viewer.is_running():
            self.close()
            return

        now = time.perf_counter()
        elapsed = min(now - self.last_wall_time, 0.1)
        self.last_wall_time = now
        self.bridge.poll()
        self._sync_operating_mode()
        if self.rules_mode and not self.hardware_imu:
            self.bridge.drain()
            self._apply_rule_state()

        commands = [float(self.controls[spec.name].get()) for spec in self.control_specs]
        if self.follow_enabled:
            commands[:-1] = self.follower.commands[:-1].tolist()
        if self.hardware_imu:
            self._process_hardware(float(self.data.time), commands)
        if self.follow_enabled:
            commands[:-1] = self.follower.commands[:-1].tolist()
            for spec, value in zip(self.control_specs, commands):
                if not spec.gripper:
                    # Disabled sliders are readouts while synchronized.
                    self.controls[spec.name].config(state="normal")
                    self.controls[spec.name].set(value)
                    self.controls[spec.name].config(state="disabled")
        self.motion.set_targets(commands)
        event = None
        send_alert = False
        with self.viewer.lock():
            self.accumulator += elapsed
            steps = int(self.accumulator / self.model.opt.timestep)
            self.accumulator -= steps * self.model.opt.timestep
            for _ in range(steps):
                self.motion.step(self.model.opt.timestep)
                mujoco.mj_step(self.model, self.data)
                if not self.hardware_imu and not self.rules_mode:
                    gyro_step = self.data.sensor("imu_gyro").data
                    lift_deg = bite_angle(self.data)
                    if self.gesture is not None:
                        detected = self.gesture.update(
                            float(self.data.time), self.data.sensor("imu_accel").data, gyro_step
                        )
                        result = (
                            self.timer.record_event(self.data.time, self.bridge.target_s)
                            if detected
                            else None
                        )
                    else:
                        result = self.timer.step(
                            self.data.time, lift_deg, gyro_step, self.bridge.target_s
                        )
                    if result is not None:
                        event = result
                        send_alert |= result == "TOO FAST"

            acc = self.data.sensor("imu_accel").data.copy()
            gyro = self.data.sensor("imu_gyro").data.copy()
            quat = self.data.sensor("imu_quat").data.copy()
            pos = self.data.sensor("wrist_pos").data.copy()
            sim_time = float(self.data.time)

        self.viewer.sync()
        # Route native viewer edits through the motion controller.
        with self.viewer.lock():
            for i, spec in enumerate(self.control_specs):
                expected = self.motion.targets[i] if spec.gripper else self.motion.position[i]
                current_ctrl = float(self.data.ctrl[self.motion.ids[i]])
                if abs(current_ctrl - expected) > 1e-6:
                    value = (
                        (current_ctrl - 0.1) / 1.4 * 100
                        if spec.gripper
                        else math.degrees(current_ctrl)
                    )
                    if not self.follow_enabled or spec.gripper:
                        self.controls[spec.name].set(value)
                    self.data.ctrl[self.motion.ids[i]] = expected
        if send_alert:
            self.bridge.alert()

        if self.rules_mode:
            ready = (
                self.bridge.live
                and self.bridge.rule_state is not None
                and self.bridge.rule_state.ready
            )
            current = self.timer.current_interval(self.bridge.latest.t) if ready else None
        elif self.hardware_imu:
            ready = self.bridge.live and self.real_imu.calibrated
            current = self.timer.current_interval(self.bridge.latest.t) if ready else None
        else:
            current = self.timer.current_interval(sim_time)
        previous = self.timer.previous_interval
        interval_status = "STARTED" if current is not None else "WAITING"
        if (self.hardware_imu or self.rules_mode) and not self.bridge.live:
            interval_status = "NO LIVE IMU"
        elif self.rules_mode and not ready:
            interval_status = "CALIBRATING"
        elif self.hardware_imu and not self.rules_mode and not self.real_imu.calibrated:
            interval_status = "CALIBRATING"
        self.metrics["Interval Status"].config(text=interval_status)
        self.metrics["Current Interval"].config(
            text=f"{current:.2f} s" if current is not None else "--"
        )
        self.metrics["Previous Interval"].config(
            text=f"{previous:.2f} s" if previous is not None else "--"
        )
        self.metrics["Pace / Target"].config(
            text=f"{self.bridge.mode_name} / {self.bridge.target_s:.1f} s"
        )
        self.details.config(
            text=f"Bites: {self.timer.bite_count}    |    Last Result: {self.timer.last_result}"
        )
        if self.rules_mode and hasattr(self, "gesture_status"):
            self.gesture_status.config(
                text="Arduino tilt rules | hold neutral still for 2 seconds after switching"
            )
        elif self.gesture is not None:
            score = float(self.gesture.scores[1]) if self.gesture.scores is not None else 0
            self.gesture_status.config(
                text=f"{self.gesture.label} | bite score {score:.0%} | simulation-trained; real-world validation pending"
            )
        if hasattr(self, "control_status"):
            label = {"PC": "PC model", "RULES": "Arduino rules (standalone)"}.get(
                self.bridge.operating_mode,
                "Legacy / unknown firmware: upload v3 for mode switching",
            )
            self.control_status.config(text=f"{label} | D4: hold 1.2 s to switch; tap for 15 s")
            self.mode_button.config(
                state="normal" if self.bridge.operating_mode != "UNKNOWN" else "disabled"
            )

        if self.hardware_imu:
            sync_status = "MANUAL / sliders enabled"
            if self.follow_enabled:
                sync_status = (
                    "SYNC PAUSED / NO LIVE IMU"
                    if not self.bridge.live
                    else self.real_imu.status
                    if not self.real_imu.calibrated
                    else self.follower.status
                )
            self.sync_label.config(
                text=sync_status, fg="#0b6e75" if sync_status.startswith("SYNC LIVE") else "#986000"
            )
            if self.bridge.live:
                sample = self.bridge.latest
                acc, gyro = sample.accel, sample.gyro
                roll, pitch = self.real_imu.roll_pitch
                attitude = (
                    f"TILT {self.real_imu.tilt:.1f} deg | roll {roll:+.1f} / pitch {pitch:+.1f} deg (estimated)"
                    if self.real_imu.calibrated
                    else self.real_imu.status
                )
                self.imu_label.config(
                    text=(
                        f"{self.bridge.chip_name} | REAL / LIVE | VQF {'REST / bias tracking' if self.real_imu.stationary else '6D'} | {sample.temperature:.1f} C\n"
                        f"ACCEL ax={acc[0]:+8.3f} ay={acc[1]:+8.3f} az={acc[2]:+8.3f} m/s^2\n"
                        f"GYRO  gx={gyro[0]:+8.3f} gy={gyro[1]:+8.3f} gz={gyro[2]:+8.3f} rad/s\n"
                        f"{attitude}"
                    )
                )
            else:
                self.imu_label.config(
                    text=(
                        f"NO LIVE HARDWARE DATA | {self.bridge.status}\n"
                        f"Sensor: {self.bridge.imu_status}\n"
                        "Waiting for real IMU samples. Gesture timing is paused.\n"
                        "If USB was unplugged, reconnect it and restart PaceLoop."
                    )
                )
            self.footer.config(
                text=(
                    f"{self.bridge.status} | "
                    + (
                        "IMU -> J1/J3/J4/J5 | Gripper manual | F9: recenter at starting pose | Yaw has no absolute reference"
                        if self.follow_enabled
                        else "Manual preview | Enable Follow forearm IMU to synchronize"
                    )
                )
            )
        else:
            self.imu_label.config(
                text=(
                    f"ACCEL    ax={acc[0]:+8.3f}  ay={acc[1]:+8.3f}  az={acc[2]:+8.3f} m/s^2\n"
                    f"GYRO     gx={gyro[0]:+8.3f}  gy={gyro[1]:+8.3f}  gz={gyro[2]:+8.3f} rad/s\n"
                    f"POSITION x ={pos[0]:+8.3f}  y ={pos[1]:+8.3f}  z ={pos[2]:+8.3f} m\n"
                    f"QUAT     w ={quat[0]:+8.3f}  x ={quat[1]:+8.3f}  "
                    f"y ={quat[2]:+8.3f}  z ={quat[3]:+8.3f}"
                )
            )
            self.footer.config(
                text=(
                    f"Arduino: {self.bridge.status} | "
                    + (
                        "Detector: learned IMU gestures | "
                        if self.gesture is not None
                        else f"Test: J4 lift > {BITE_ANGLE_DEG:.0f} deg + gyro; return < {RESET_ANGLE_DEG:.0f} deg | "
                    )
                    + f"Sim {sim_time:.1f}s | speed {sim_time / max(now - self.wall_start, 0.001):.2f}x"
                )
            )

            for i, value in enumerate([*acc, *gyro]):
                self.history[i].append(float(value))
            self._log_sample(
                sim_time,
                commands,
                acc,
                gyro,
                quat,
                pos,
                event,
                sim_time,
                "simulation",
                bite_angle(self.data),
                "VIRTUAL",
            )
            self.bridge.drain()
        self._draw_graph()
        if time.monotonic() - self.last_flush >= 1:
            self.csv_file.flush()
            self.last_flush = time.monotonic()
        self.root.after(10, self.tick)

    def close(self):
        if not self.running:
            return
        self.running = False
        try:
            self.viewer.close()
        finally:
            self.bridge.close()
            self.csv_file.close()
            self.root.destroy()
        print("Saved:", self.csv_path)

    def run(self):
        try:
            self.root.mainloop()
        finally:
            if self.running:
                self.close()


def main():
    parser = argparse.ArgumentParser(description="OpenArm PaceLoop + Arduino Bridge")
    parser.add_argument(
        "--port", default=None, help="Arduino port, e.g. COM3; omit to run without Arduino"
    )
    parser.add_argument(
        "--csv", type=Path, help="New CSV path (existing files are never overwritten)"
    )
    parser.add_argument(
        "--imu-source",
        choices=("auto", "hardware", "sim"),
        default="auto",
        help="auto: real IMU when --port is supplied, otherwise virtual IMU",
    )
    parser.add_argument(
        "--manual", action="store_true", help="Start hardware mode with IMU arm following disabled"
    )
    parser.add_argument("--gesture-model", type=Path, help="Optional trained IMU classifier (.npz)")
    parser.add_argument(
        "--control-mode",
        choices=("pc", "rules"),
        default="pc",
        help="Initial Arduino mode; D4 long press switches while running (default: pc)",
    )
    parser.add_argument(
        "--gesture-demo", action="store_true", help="Open the labeled gesture replay viewer"
    )
    args = parser.parse_args()
    if args.gesture_demo:
        if args.port:
            parser.error("The gesture demo uses simulated data; omit --port")
        from paceloop.app.viewer import GestureViewer

        GestureViewer(args.gesture_model or ROOT / "models" / DEFAULT_MODEL_FILE).run()
        return
    source = ("hardware" if args.port else "sim") if args.imu_source == "auto" else args.imu_source
    if source == "hardware" and not args.port:
        parser.error("--imu-source hardware requires --port")
    if args.control_mode == "rules" and not args.port:
        parser.error("--control-mode rules requires --port; standalone use needs only board power")
    if source == "hardware" and VQF is None:
        parser.error("Real IMU requires VQF: py -3.11 -m pip install vqf==2.1.2")

    try:
        model_path = args.gesture_model or (
            ROOT / "models" / DEFAULT_MODEL_FILE if args.port else None
        )
        gesture_model = TinyGestureModel.load(model_path) if model_path else None
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"Cannot load gesture model: {exc}")
    model, data = load_model(ROBOT_XML)
    bridge = ArduinoBridge(args.port, control_mode=args.control_mode.upper() if args.port else None)
    app = PaceLoopApp(
        model,
        data,
        bridge,
        args.csv,
        source,
        follow_imu=not args.manual,
        gesture_model=gesture_model,
    )
    app.run()


if __name__ == "__main__":
    main()
