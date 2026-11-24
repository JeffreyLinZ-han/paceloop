"""Run student inference on IMU recordings and save predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from paceloop.learning.data import load_imu, recording_features, sha256
from paceloop.learning.model import LABELS, TinyGestureModel


def predict(model_path, input_dir, output):
    model_path, input_dir, output = Path(model_path), Path(input_dir), Path(output)
    model = TinyGestureModel.load(model_path)
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise ValueError("No student IMU recordings found")
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        format="student_predictions_v1",
        model_sha256=sha256(model_path),
        labels=list(LABELS),
        threshold=model.threshold,
        class_weights=model.metadata.get("class_weights"),
        student_fields=["time", "signal"],
        teacher_information_loaded=False,
        episodes=[],
    )
    for path in files:
        t, signal = load_imu(path)
        ends, features = recording_features(t, signal)
        scores = model.probabilities(features)
        np.savez_compressed(
            output / path.name, time=t[ends], scores=scores, input_sha256=sha256(path)
        )
        manifest["episodes"].append(path.name)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Student saved predictions for {len(files)} IMU-only recordings")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="IMU-only directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predict(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
