"""Interactive OpenArm gesture replay with live six-axis classifier scores."""

import argparse
import csv
import os
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import mujoco
import mujoco.viewer
import numpy as np

from paceloop.learning.metrics import cross_entropy
from paceloop.learning.model import (
    DEFAULT_MODEL_FILE,
    LABELS,
    SAMPLE_HZ,
    GestureRecognizer,
    TinyGestureModel,
)
from paceloop.paths import ROOT
from paceloop.training.motion import GestureSimulator

NAMES = {
    "other": "Other / rest",
    "bite": "Hand to mouth",
    "scratch_head": "Scratch head",
    "touch_face": "Touch face",
    "warming_up": "Collecting IMU window",
}


class GestureViewer:
    def __init__(self, model_path, smoke_seconds=None):
        self.model_path = Path(model_path)
        self.classifier = TinyGestureModel.load(model_path)
        self.recognizer = GestureRecognizer(self.classifier)
        self.simulator = GestureSimulator()
        self.model, self.data = self.simulator.model, self.simulator.data
        self.running = True
        self.training_window = None
        self.playing = True
        self.variant = 0
        self.action = 1
        self.root = tk.Tk()
        self.root.title("PaceLoop | Gesture Viewer")
        panel_height = min(840, self.root.winfo_screenheight() - 100)
        self.root.geometry(f"470x{panel_height}+15+35")
        self.root.minsize(440, 780)
        self.root.option_add("*Font", ("Segoe UI", 10))
        self.root.configure(bg="#f2f5f8")
        self.speed = tk.DoubleVar(value=1.0)
        self.cycle = tk.BooleanVar(value=True)
        self.show_teacher = tk.BooleanVar(value=False)
        self.status = tk.StringVar()
        self.truth = tk.StringVar()
        self.prediction = tk.StringVar(value="Collecting IMU window")
        self.event_text = tk.StringVar()
        self.loss_text = tk.StringVar()
        self._controls()
        self.load_episode(1)
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        )
        with self.viewer.lock():
            self.viewer.cam.lookat[:] = (0.10, 0, 0.58)
            self.viewer.cam.distance = 1.65
            self.viewer.cam.azimuth = 145
            self.viewer.cam.elevation = -12
            self.viewer.opt.geomgroup[3] = 0
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<space>", lambda event: self.toggle_play())
        self.root.bind("<Left>", lambda event: self.load_episode((self.action - 1) % 4))
        self.root.bind("<Right>", lambda event: self.load_episode((self.action + 1) % 4))
        self.root.after(300, self._arrange_windows)
        self.root.after(16, self.tick)
        if smoke_seconds is not None:
            self.root.after(int(smoke_seconds * 1000), self.close)

    def _controls(self):
        title = tk.Frame(self.root, bg="#123247", padx=16, pady=12)
        title.pack(fill="x")
        tk.Label(
            title,
            text="PaceLoop | Gesture Viewer",
            bg="#123247",
            fg="white",
            font=("Segoe UI", 17, "bold"),
        ).pack(anchor="w")
        tk.Label(title, text="Teacher replay | Student: IMU only", bg="#123247", fg="#bde5e8").pack(
            anchor="w"
        )
        actions = tk.LabelFrame(self.root, text="Choose a gesture", padx=8, pady=8)
        actions.pack(fill="x", padx=12, pady=10)
        for i in (1, 2, 3, 0):
            tk.Button(
                actions,
                text=NAMES[LABELS[i]],
                command=lambda label=i: self.load_episode(label),
                width=18,
            ).grid(row=(i - 1) % 4 // 2, column=(i - 1) % 2, padx=5, pady=4, sticky="ew")
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        playback = tk.Frame(self.root)
        playback.pack(fill="x", padx=12)
        self.play_button = tk.Button(playback, text="Pause [Space]", command=self.toggle_play)
        self.play_button.pack(side="left")
        tk.Button(playback, text="Replay", command=self.replay).pack(side="left", padx=6)
        tk.Button(playback, text="New variation", command=self.new_variant).pack(side="left")
        tk.Checkbutton(playback, text="Cycle", variable=self.cycle).pack(side="right")
        tk.Scale(
            self.root,
            from_=0.25,
            to=2.0,
            resolution=0.25,
            orient="horizontal",
            variable=self.speed,
            label="Playback speed (sensor timestamps stay fixed)",
        ).pack(fill="x", padx=14, pady=4)
        result = tk.LabelFrame(self.root, text="Recognition", padx=12, pady=8)
        result.pack(fill="x", padx=12, pady=6)
        evaluation = tk.Frame(result)
        evaluation.pack(fill="x")
        tk.Checkbutton(evaluation, text="Show teacher labels", variable=self.show_teacher).pack(
            side="left"
        )
        tk.Button(evaluation, text="Training loss", command=self.show_training_loss).pack(
            side="right"
        )
        tk.Label(result, textvariable=self.truth, anchor="w", justify="left", wraplength=400).pack(
            fill="x"
        )
        self.result_label = tk.Label(
            result,
            textvariable=self.prediction,
            anchor="w",
            font=("Segoe UI", 13, "bold"),
            fg="#087e8b",
        )
        self.result_label.pack(fill="x", pady=5)
        self.bars, self.score_labels = {}, {}
        for label in LABELS:
            row = tk.Frame(result)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=NAMES[label], width=13, anchor="w").pack(side="left")
            bar = ttk.Progressbar(row, maximum=1, length=175)
            bar.pack(side="left", fill="x", expand=True)
            score = tk.Label(row, text="--", width=5, anchor="e")
            score.pack(side="right")
            self.bars[label], self.score_labels[label] = bar, score
        tk.Label(result, textvariable=self.event_text, anchor="w", fg="#b05c13").pack(
            fill="x", pady=(8, 0)
        )
        tk.Label(result, textvariable=self.loss_text, anchor="w", fg="#526574").pack(
            fill="x", pady=3
        )
        self.timeline = ttk.Progressbar(self.root, maximum=1)
        self.timeline.pack(fill="x", padx=15, pady=(7, 2))
        tk.Label(self.root, textvariable=self.status, bg="#f2f5f8", anchor="w").pack(
            fill="x", padx=15
        )
        tk.Label(
            self.root, text="Raw IMU: acceleration / angular velocity", bg="#f2f5f8", anchor="w"
        ).pack(fill="x", padx=15, pady=(6, 0))
        self.canvas = tk.Canvas(self.root, height=100, bg="#122331", highlightthickness=0)
        tk.Label(
            self.root,
            text="Teacher labels are used only for scoring, never as inputs.\nSimulation only | Fixed student weights during replay.",
            bg="#f2f5f8",
            fg="#526574",
            justify="left",
            anchor="w",
        ).pack(side="bottom", fill="x", padx=15, pady=6)
        self.canvas.pack(fill="both", expand=True, padx=14, pady=4)

    def load_episode(self, action):
        self.action = action
        self.episode = self.simulator.episode(50000 + self.variant, self.variant, action)
        self.velocity = np.gradient(self.episode["joints"], 1 / SAMPLE_HZ, axis=0, edge_order=2)
        self.acceleration = np.gradient(self.velocity, 1 / SAMPLE_HZ, axis=0, edge_order=2)
        self.replay()

    def replay(self):
        self.recognizer.reset()
        self.index = -1
        self.elapsed = 0.0
        self.event_count = 0
        self.last_event = None
        self.loss_sum = 0.0
        self.loss_windows = 0
        self.last_loss = None
        self.loss_text.set("Cross-entropy: waiting for an IMU window")
        self.last_wall = time.perf_counter()
        self.playing = True
        self.play_button.config(text="Pause [Space]")
        self.event_text.set("Bite events this replay: 0")
        self.prediction.set("Prediction: collecting IMU window")
        for label in LABELS:
            self.bars[label]["value"] = 0
            self.score_labels[label].config(text="--")

    def new_variant(self):
        self.variant += 1
        self.load_episode(self.action)

    def toggle_play(self):
        self.playing = not self.playing
        self.last_wall = time.perf_counter()
        self.play_button.config(text="Pause [Space]" if self.playing else "Play [Space]")

    def show_training_loss(self):
        if self.training_window is not None and self.training_window.winfo_exists():
            self.training_window.lift()
            return
        history_path = self.model_path.with_suffix(".history.csv")
        window = tk.Toplevel(self.root)
        self.training_window = window
        window.title("PaceLoop | Student training loss")
        window.geometry(f"720x440+550+{max(50, self.root.winfo_screenheight() - 560)}")
        if not history_path.exists():
            tk.Label(window, text="This checkpoint has no recorded training history.").pack(
                padx=20, pady=20
            )
            return
        with history_path.open(newline="", encoding="utf-8") as stream:
            history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        best_epoch = int(self.classifier.metadata["best_epoch"])
        selected = history[best_epoch - 1]
        tk.Label(
            window,
            text="Student training: cross-entropy loss (lower is better)",
            font=("Segoe UI", 13, "bold"),
        ).pack(pady=(12, 4))
        tk.Label(
            window,
            text=f"Selected epoch {best_epoch} | Train {selected['train_ce']:.4f} | "
            f"Validation {selected['validation_ce']:.4f}",
        ).pack()
        canvas = tk.Canvas(window, bg="white", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=12, pady=10)
        detail = tk.StringVar(value="Blue: train | Orange: validation | Dashed: selected epoch")
        tk.Label(window, textvariable=detail).pack(pady=(0, 8))
        tk.Label(
            window,
            text="Curves show unweighted CE. Selection uses class-weighted validation CE.",
            fg="#526574",
        ).pack(pady=(0, 10))

        def draw(_=None):
            canvas.delete("all")
            width, height = max(300, canvas.winfo_width()), max(150, canvas.winfo_height())
            left, top, right, bottom = 55, 20, width - 20, height - 35
            maximum = (
                max(row[key] for row in history for key in ("train_ce", "validation_ce")) * 1.1
            )
            for tick in range(5):
                value = maximum * tick / 4
                y = bottom - (bottom - top) * tick / 4
                canvas.create_line(left, y, right, y, fill="#e2e8ee")
                canvas.create_text(left - 8, y, text=f"{value:.2f}", anchor="e")
            for key, color in (("train_ce", "#167db2"), ("validation_ce", "#db7b23")):
                points = []
                for row in history:
                    points.extend(
                        (
                            left + (row["epoch"] - 1) * (right - left) / max(1, len(history) - 1),
                            bottom - row[key] * (bottom - top) / max(maximum, 1e-9),
                        )
                    )
                if len(points) >= 4:
                    canvas.create_line(*points, fill=color, width=2)
            x = left + (best_epoch - 1) * (right - left) / max(1, len(history) - 1)
            canvas.create_line(x, top, x, bottom, fill="#748594", dash=(4, 4))
            canvas.create_text(left, bottom + 15, text="1")
            canvas.create_text(right, bottom + 15, text=str(len(history)))
            canvas.create_text((left + right) / 2, bottom + 18, text="Epoch")

        def hover(event):
            span = max(1, canvas.winfo_width() - 75)
            index = int(
                np.clip(round((event.x - 55) * (len(history) - 1) / span), 0, len(history) - 1)
            )
            row = history[index]
            detail.set(
                f"Epoch {int(row['epoch'])} | Train CE {row['train_ce']:.4f} | "
                f"Validation CE {row['validation_ce']:.4f}"
            )

        canvas.bind("<Configure>", draw)
        canvas.bind("<Motion>", hover)

    def _arrange_windows(self):
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            user32.GetWindowThreadProcessId.argtypes = [
                wintypes.HWND,
                ctypes.POINTER(wintypes.DWORD),
            ]
            user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
            user32.MoveWindow.argtypes = [
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.BOOL,
            ]

            @callback_type
            def arrange(hwnd, _):
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                title = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, title, 256)
                if pid.value == os.getpid() and "MuJoCo" in title.value:
                    user32.MoveWindow(
                        hwnd,
                        520,
                        35,
                        max(640, self.root.winfo_screenwidth() - 550),
                        min(850, self.root.winfo_screenheight() - 90),
                        True,
                    )
                return True

            user32.EnumWindows(arrange, 0)
        self.root.lift()

    def _scene(self, index):
        with self.viewer.lock():
            self.data.qpos[self.simulator.qids] = self.episode["joints"][index]
            self.data.qvel[:] = 0
            self.data.qacc[:] = 0
            self.data.qvel[self.simulator.vids] = self.velocity[index]
            self.data.qacc[self.simulator.vids] = self.acceleration[index]
            self.data.time = float(self.episode["time"][index])
            mujoco.mj_inverse(self.model, self.data)
            # Render-only head marker.
            scene = self.viewer.user_scn
            scene.ngeom = 1
            head_center = self.data.body("openarm_body_link0").xpos + np.array([0, 0, 0.875])
            material = self.model.geom_matid[self.model.geom("openarm_body_link0_visual_0").id]
            mujoco.mjv_initGeom(
                scene.geoms[0],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.115, 0, 0]),
                head_center,
                np.eye(3).ravel(),
                self.model.mat_rgba[material],
            )
            scene.geoms[0].specular = self.model.mat_specular[material]
            scene.geoms[0].shininess = self.model.mat_shininess[material]
            scene.geoms[0].reflectance = self.model.mat_reflectance[material]
        self.viewer.sync()

    def _plot(self, index):
        self.canvas.delete("all")
        values = self.episode["signal"][max(0, index - 119) : index + 1]
        w, h = max(100, self.canvas.winfo_width()), max(40, self.canvas.winfo_height())
        colors = ("#ff8377", "#ffdb77", "#5de2c0", "#74bfff", "#d7a5ff", "#f5a969")
        if len(values) < 2:
            return
        for channel, color in enumerate(colors):
            scale = 18 if channel < 3 else 3
            center = h * (0.25 if channel < 3 else 0.75)
            points = []
            for i, value in enumerate(values[:, channel]):
                points.extend(
                    (
                        5 + i * (w - 10) / 119,
                        center - float(np.clip(value / scale, -1, 1)) * h * 0.2,
                    )
                )
            self.canvas.create_line(*points, fill=color, width=1.5)

    def tick(self):
        if not self.running:
            return
        if not self.viewer.is_running():
            self.close()
            return
        now = time.perf_counter()
        if self.playing:
            self.elapsed += min(now - self.last_wall, 0.2) * self.speed.get()
        self.last_wall = now
        times = self.episode["time"]
        target = min(len(times) - 1, int(np.searchsorted(times, self.elapsed, side="right")) - 1)
        while self.index < target:
            self.index += 1
            sample = self.episode["signal"][self.index]
            previous_count = self.recognizer.prediction_count
            if self.recognizer.update(float(times[self.index]), sample[:3], sample[3:]):
                self.event_count += 1
                self.last_event = float(times[self.index])
            if self.recognizer.prediction_count != previous_count:
                # Score the completed prediction against the teacher label.
                scored_index = (
                    int(
                        np.searchsorted(times, self.recognizer.prediction_time + 1e-8, side="right")
                    )
                    - 1
                )
                label = int(self.episode["labels"][scored_index])
                self.last_loss = cross_entropy(np.array([label]), self.recognizer.scores[None, :])
                self.loss_sum += self.last_loss
                self.loss_windows += 1
        if self.index >= 0:
            self._scene(self.index)
            truth = LABELS[int(self.episode["labels"][self.index])]
            self.truth.set(
                f"Teacher: {NAMES[LABELS[self.action]]} | Now: {NAMES[truth]}"
                if self.show_teacher.get()
                else "Teacher labels hidden from display"
            )
            self.prediction.set("Prediction: " + NAMES[self.recognizer.label])
            match = self.recognizer.label == truth
            self.result_label.config(
                fg="#526574"
                if self.recognizer.scores is None
                else "#087e8b"
                if match
                else "#b05c13"
            )
            if self.recognizer.scores is not None:
                for label, score in zip(LABELS, self.recognizer.scores):
                    self.bars[label]["value"] = float(score)
                    self.score_labels[label].config(text=f"{score:.0%}")
            suffix = f"(last at {self.last_event:.1f}s)" if self.last_event is not None else ""
            self.event_text.set(f"Bite events this replay: {self.event_count} {suffix}")
            if self.last_loss is not None:
                self.loss_text.set(
                    f"Cross-entropy: {self.last_loss:.3f} | Mean: "
                    f"{self.loss_sum / self.loss_windows:.3f} ({self.loss_windows} windows)"
                )
            self.timeline["value"] = min(1, self.elapsed / times[-1])
            self.status.set(
                f"{min(self.elapsed, times[-1]):.1f} / {times[-1]:.1f} s  |  Variation {self.variant + 1}"
            )
            self._plot(self.index)
        if self.playing and self.elapsed > times[-1] + 1.2:
            if self.cycle.get():
                self.load_episode((self.action + 1) % 4)
            else:
                self.toggle_play()
        self.root.after(30, self.tick)

    def close(self):
        if not self.running:
            return
        self.running = False
        self.viewer.close()
        self.root.destroy()

    def run(self):
        try:
            self.root.mainloop()
        finally:
            if self.running:
                self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / DEFAULT_MODEL_FILE)
    parser.add_argument("--smoke-seconds", type=float, help="Close automatically after a GUI check")
    parser.add_argument(
        "--show-loss", action="store_true", help="Open the saved training loss curves"
    )
    args = parser.parse_args()
    app = GestureViewer(args.model, args.smoke_seconds)
    if args.show_loss:
        app.root.after(600, app.show_training_loss)
    app.run()


if __name__ == "__main__":
    main()
