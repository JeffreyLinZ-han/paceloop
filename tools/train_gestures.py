"""Train a 208 -> 32 -> 4 MLP and evaluate independent simulated sessions."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from gesture_data import load_imu, recording_features
from gesture_metrics import classification_metrics, cross_entropy, event_metrics
from gesture_model import (
    FEATURE_VERSION,
    LABELS,
    SAMPLE_HZ,
    WINDOW_SAMPLES,
    TinyGestureModel,
)

ROOT = Path(__file__).resolve().parents[1]


def load_split(folder, manifest, split):
    features, targets, episodes = [], [], []
    offset = 0
    for item in manifest["episodes"]:
        if item["split"] != split:
            continue
        separated = manifest.get("format") == "teacher_student_v1"
        target = folder / "teacher" / split / item["file"] if separated else folder / item["file"]
        with np.load(target, allow_pickle=False) as data:
            labels = data["labels"]
            if separated:
                t, signal = load_imu(folder / "imu" / split / item["file"])
                if not np.array_equal(t, data["time"]):
                    raise ValueError("Sensor and teacher timestamps differ")
            else:
                signal, t = data["signal"], data["time"]
            ends, values = recording_features(t, signal)
            features.append(values)
            targets.append(labels[ends])
            episodes.append(
                dict(
                    file=item["file"],
                    session=item["session"],
                    action=int(data["action"]),
                    start=offset,
                    stop=offset + len(ends),
                    time=t[ends],
                    interval=data["event_interval"].tolist(),
                )
            )
            offset += len(ends)
    return np.concatenate(features), np.concatenate(targets), episodes


def train(train_x, train_y, val_x, val_y, epochs=180):
    rng = np.random.default_rng(3107)
    mean, scale = train_x.mean(axis=0), np.maximum(train_x.std(axis=0), 0.03)
    x = np.clip((train_x - mean) / scale, -8, 8).astype(np.float32)
    vx = np.clip((val_x - mean) / scale, -8, 8).astype(np.float32)
    params = [
        rng.normal(0, np.sqrt(2 / 208), (208, 32)).astype(np.float32),
        np.zeros(32, np.float32),
        rng.normal(0, np.sqrt(2 / 32), (32, 4)).astype(np.float32),
        np.zeros(4, np.float32),
    ]
    first, second = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
    counts = np.bincount(train_y, minlength=4)
    class_weight = np.sqrt(len(train_y) / (4 * np.maximum(counts, 1)))
    best, best_loss, patience, iteration, best_epoch = None, float("inf"), 0, 0, 0
    history = []

    def forward(data):
        hidden = np.maximum(data @ params[0] + params[1], 0)
        logits = hidden @ params[2] + params[3]
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        return hidden, probs / probs.sum(axis=1, keepdims=True)

    for epoch in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 256):
            idx = order[start : start + 256]
            batch, y = x[idx], train_y[idx]
            hidden, probs = forward(batch)
            probs[np.arange(len(idx)), y] -= 1
            probs *= (class_weight[y] / class_weight[y].sum())[:, None]
            dh = (probs @ params[2].T) * (hidden > 0)
            gradients = [
                batch.T @ dh + 0.0001 * params[0],
                dh.sum(axis=0),
                hidden.T @ probs + 0.0001 * params[2],
                probs.sum(axis=0),
            ]
            iteration += 1
            for i, (p, grad) in enumerate(zip(params, gradients)):
                first[i] = 0.9 * first[i] + 0.1 * grad
                second[i] = 0.999 * second[i] + 0.001 * grad * grad
                p -= (
                    0.001
                    * (first[i] / (1 - 0.9**iteration))
                    / (np.sqrt(second[i] / (1 - 0.999**iteration)) + 1e-8)
                )
        _, vp = forward(vx)
        _, tp = forward(x)
        loss = cross_entropy(val_y, vp, class_weight)
        train_weighted = cross_entropy(train_y, tp, class_weight)
        penalty = 0.5 * 0.0001 * (np.sum(params[0] ** 2) + np.sum(params[2] ** 2))
        history.append(
            dict(
                epoch=epoch + 1,
                train_ce=cross_entropy(train_y, tp),
                validation_ce=cross_entropy(val_y, vp),
                train_weighted_ce=train_weighted,
                validation_weighted_ce=loss,
                train_objective=float(train_weighted + penalty),
            )
        )
        if loss < best_loss - 0.0001:
            best, best_loss, patience, best_epoch = [p.copy() for p in params], loss, 0, epoch + 1
        else:
            patience += 1
        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}: train CE {history[-1]['train_ce']:.4f}, "
                f"validation CE {history[-1]['validation_ce']:.4f}",
                flush=True,
            )
        if patience >= 25:
            break
    metadata = dict(
        feature_version=FEATURE_VERSION,
        labels=list(LABELS),
        sample_hz=SAMPLE_HZ,
        window_samples=WINDOW_SAMPLES,
        source="simulation_only",
        real_world_validated=False,
        architecture=[208, 32, 4],
        threshold=0.8,
        best_epoch=best_epoch,
        units=["m/s^2"] * 3 + ["rad/s"] * 3,
        class_weights=class_weight.tolist(),
        supervision="scripted_teacher_labels",
        student_inputs=["ax", "ay", "az", "gx", "gy", "gz"],
        teacher_fields_used_as_features=False,
    )
    return TinyGestureModel(mean, scale, *best, metadata), history


def run(folder, output, epochs):
    folder, output = Path(folder), Path(output)
    if any(
        output.with_suffix(suffix).exists()
        for suffix in (".npz", ".metrics.json", ".history.csv", ".predictions")
    ):
        raise ValueError("Choose a new model filename; existing training results are preserved")
    manifest_bytes = (folder / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    groups = {
        s: {e["session"] for e in manifest["episodes"] if e["split"] == s}
        for s in ("train", "validation", "test")
    }
    if not all(groups.values()) or any(
        groups[a] & groups[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Training, validation and test sessions must be nonempty and disjoint")
    tx, ty, te = load_split(folder, manifest, "train")
    vx, vy, ve = load_split(folder, manifest, "validation")
    print(
        f"Training on {len(te)} episodes / {len(tx)} windows; validation {len(ve)} episodes",
        flush=True,
    )
    started = time.perf_counter()
    model, history = train(tx, ty, vx, vy, epochs)
    validation_scores = model.probabilities(vx)
    candidates = [
        (float(threshold), event_metrics(ve, validation_scores, threshold))
        for threshold in np.arange(0.55, 0.96, 0.05)
    ]
    threshold, validation_events = max(
        candidates, key=lambda item: (item[1]["f1"], item[1]["precision"], item[0])
    )
    model.threshold = threshold
    model.metadata.update(
        threshold=threshold, dataset_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest()
    )
    model.save(output)
    with output.with_suffix(".history.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    # Test is opened only after model selection and event threshold selection.
    if manifest.get("format") == "teacher_student_v1":
        from tools.evaluate_gestures import evaluate
        from tools.predict_gestures import predict

        prediction_dir = output.with_suffix(".predictions")
        predict(output, folder / "imu" / "test", prediction_dir)
        blind_report = evaluate(prediction_dir, folder / "teacher" / "test")
        output.with_suffix(".blind_test.json").write_text(
            json.dumps(blind_report, indent=2) + "\n", encoding="utf-8"
        )
        test_metrics, test_events = blind_report["classification"], blind_report["events"]
        test_episodes, test_windows = blind_report["episodes"], blind_report["windows"]
    else:
        ex, ey, ee = load_split(folder, manifest, "test")
        scores = model.probabilities(ex)
        test_metrics, test_events = (
            classification_metrics(ey, scores),
            event_metrics(ee, scores, threshold),
        )
        test_episodes, test_windows = len(ee), len(ex)
    report = dict(
        source="simulation_only",
        real_world_validated=False,
        labels=list(LABELS),
        architecture=model.metadata["architecture"],
        trainable_parameters=int(sum(a.size for a in model.arrays[2:])),
        split_sessions={k: sorted(v) for k, v in groups.items()},
        episodes={"train": len(te), "validation": len(ve), "test": test_episodes},
        windows={"train": len(tx), "validation": len(vx), "test": test_windows},
        validation=classification_metrics(vy, validation_scores),
        validation_events=validation_events,
        test=test_metrics,
        test_events=test_events,
        training_history=history,
        selected_epoch_losses=history[model.metadata["best_epoch"] - 1],
        supervision=model.metadata["supervision"],
        student_inputs=model.metadata["student_inputs"],
        loss_definition="Mean negative natural log probability of the teacher class",
        selection_metric="validation_weighted_ce",
        blind_test=manifest.get("format") == "teacher_student_v1",
        threshold=threshold,
        best_epoch=model.metadata["best_epoch"],
        training_seconds=time.perf_counter() - started,
        dataset_manifest_sha256=model.metadata["dataset_manifest_sha256"],
        limitations=[
            "Procedural robot motion, not human eating recordings",
            "No observation of food, utensil contact or swallowing",
            "Identical wrist trajectories cannot reveal intent",
            "Real sensor mounting and human movement require labeled validation",
        ],
    )
    report["model_bytes"] = output.stat().st_size
    report["model_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("trainable_parameters", "model_bytes", "test_events", "threshold")
            },
            indent=2,
        )
    )
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets" / "gesture_student_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "gesture_student_v2.npz")
    parser.add_argument("--epochs", type=int, default=180)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    run(args.data, args.output, args.epochs)


if __name__ == "__main__":
    main()
