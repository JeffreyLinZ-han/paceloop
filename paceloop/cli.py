"""Terminal commands for the PaceLoop twin, training, and gesture viewer."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from paceloop.paths import ROOT

LAST_MODEL = ROOT / "logs" / "last_trained_model.json"


def selected_model():
    if LAST_MODEL.is_file():
        try:
            path = Path(json.loads(LAST_MODEL.read_text(encoding="utf-8"))["model"])
            if path.is_file():
                return path
        except (OSError, ValueError, KeyError, TypeError):
            pass
    from paceloop.learning.model import DEFAULT_MODEL_FILE

    return ROOT / "models" / DEFAULT_MODEL_FILE


def has_option(arguments, *names):
    return any(value.split("=", 1)[0] in names for value in arguments)


def forward(command, arguments):
    help_requested = has_option(arguments, "-h", "--help")
    if command == "twin":
        from paceloop.app.twin import main

        arguments = [
            value.replace("--model", "--gesture-model", 1)
            if value == "--model" or value.startswith("--model=")
            else value
            for value in arguments
        ]
        flag = "--gesture-model"
    else:
        from paceloop.app.viewer import main

        flag = "--model"
    if not help_requested and not has_option(arguments, flag):
        path = selected_model()
        arguments = [*arguments, flag, str(path)]
        print(f"Model: {path}", flush=True)
    previous = sys.argv
    try:
        sys.argv = [f"loop {command}", *arguments]
        main()
    finally:
        sys.argv = previous


def train(arguments):
    parser = argparse.ArgumentParser(prog="loop train", description="Train the IMU gesture model.")
    parser.add_argument("--data", type=Path, default=ROOT / "datasets" / "gesture_student_v2")
    parser.add_argument("--output", type=Path, help="New .npz file (default: a timestamped run)")
    parser.add_argument("--epochs", type=int, default=180, help="Maximum epochs (default: 180)")
    args = parser.parse_args(arguments)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not (args.data / "manifest.json").is_file():
        parser.error(
            f"Training data missing: {args.data}. See docs/gesture-model.md to generate it."
        )
    output = args.output or ROOT / "models" / "runs" / (
        "gesture_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".npz"
    )
    if output.suffix.lower() != ".npz":
        parser.error("--output must end in .npz")
    from paceloop.training.train import run

    print(f"Data: {args.data}\nOutput: {output}\nPress Ctrl+C to stop training.", flush=True)
    run(args.data, output, args.epochs)
    LAST_MODEL.parent.mkdir(parents=True, exist_ok=True)
    LAST_MODEL.write_text(json.dumps({"model": str(output.resolve())}) + "\n", encoding="utf-8")
    print("Training complete. 'loop twin' and 'loop show' will use this model.")


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="loop",
        description="PaceLoop: simulation, IMU training, and gesture replay.",
        epilog="Use 'loop <command> --help' for options. Press Ctrl+C to stop.",
    )
    parser.add_argument("command", choices=("twin", "train", "show"), help="What to open or run")
    if not arguments:
        parser.print_help()
        return 0
    command = parser.parse_args(arguments[:1]).command
    try:
        if command == "train":
            train(arguments[1:])
        else:
            forward(command, arguments[1:])
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except ModuleNotFoundError as exc:
        parser.exit(1, f"Missing dependency: {exc.name}. Install requirements.txt in .venv.\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
