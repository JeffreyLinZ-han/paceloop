"""Score saved student predictions against teacher labels."""

import argparse
import json
from pathlib import Path

import numpy as np

from paceloop.learning.metrics import classification_metrics, cross_entropy, event_metrics
from paceloop.learning.model import STRIDE_SAMPLES, WINDOW_SAMPLES


def evaluate(prediction_dir, teacher_dir):
    prediction_dir, teacher_dir = Path(prediction_dir), Path(teacher_dir)
    manifest = json.loads((prediction_dir / "manifest.json").read_text(encoding="utf-8"))
    scores, truth, episodes = [], [], []
    offset = 0
    for filename in manifest["episodes"]:
        if Path(filename).name != filename:
            raise ValueError("Invalid prediction filename")
        with (
            np.load(prediction_dir / filename, allow_pickle=False) as prediction,
            np.load(teacher_dir / filename, allow_pickle=False) as teacher,
        ):
            if str(prediction["input_sha256"].item()) != str(teacher["input_sha256"].item()):
                raise ValueError("Prediction and teacher refer to different sensor recordings")
            ends = np.arange(WINDOW_SAMPLES - 1, len(teacher["time"]), STRIDE_SAMPLES)
            if prediction["time"].shape != ends.shape or not np.allclose(
                prediction["time"], teacher["time"][ends], atol=1e-6, rtol=0
            ):
                raise ValueError("Prediction and teacher timestamps do not match")
            p, y = prediction["scores"], teacher["labels"][ends]
            cross_entropy(y, p)
            scores.append(p)
            truth.append(y)
            episodes.append(
                dict(
                    start=offset,
                    stop=offset + len(ends),
                    time=prediction["time"],
                    action=int(teacher["action"]),
                    interval=teacher["event_interval"].tolist(),
                )
            )
            offset += len(ends)
    if not episodes:
        raise ValueError("No predictions to evaluate")
    p, y = np.concatenate(scores), np.concatenate(truth)
    metrics = classification_metrics(y, p)
    if manifest.get("class_weights") is not None:
        metrics["weighted_cross_entropy"] = cross_entropy(y, p, manifest["class_weights"])
    return dict(
        source="simulation_only",
        student_fields=manifest["student_fields"],
        teacher_information_loaded_by_student=manifest["teacher_information_loaded"],
        model_sha256=manifest["model_sha256"],
        model_updated=False,
        episodes=len(episodes),
        windows=len(y),
        classification=metrics,
        events=event_metrics(episodes, p, manifest["threshold"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.predictions, args.teacher)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["classification"], indent=2))


if __name__ == "__main__":
    main()
