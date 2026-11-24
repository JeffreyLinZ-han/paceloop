"""Separate scripted recordings into student IMU files and teacher-only targets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gesture_data import sha256

ROOT = Path(__file__).resolve().parents[1]


def prepare(source, output):
    source, output = Path(source), Path(output)
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        format="teacher_student_v1",
        source_manifest_sha256=sha256(source / "manifest.json"),
        student_fields=["time", "signal"],
        student_channels=["ax", "ay", "az", "gx", "gy", "gz"],
        source="simulation_only",
        episodes=[],
    )
    for item in source_manifest["episodes"]:
        split = item["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown dataset split")
        with np.load(source / item["file"], allow_pickle=False) as data:
            # The neutral ID is derived solely from sensor bytes, not the action.
            identifier = hashlib.sha256(
                data["signal"].tobytes() + data["time"].tobytes()
            ).hexdigest()[:20]
            filename = identifier + ".npz"
            sensor_path, teacher_path = (
                output / "imu" / split / filename,
                output / "teacher" / split / filename,
            )
            sensor_path.parent.mkdir(parents=True, exist_ok=True)
            teacher_path.parent.mkdir(parents=True, exist_ok=True)
            if sensor_path.exists():
                raise ValueError("Duplicate sensor episode")
            np.savez_compressed(sensor_path, time=data["time"], signal=data["signal"])
            privileged = {k: data[k] for k in data.files if k != "signal"}
            np.savez_compressed(teacher_path, **privileged, input_sha256=sha256(sensor_path))
            manifest["episodes"].append(dict(file=filename, split=split, session=item["session"]))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Separated {len(manifest['episodes'])} recordings into imu/ and teacher/")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets" / "gestures_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "gesture_student_v2")
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
