# Changelog

## Repository organization

- Grouped application, training, network, bridge, and simulation code under
  `paceloop/`, with shared resource paths in `paceloop/paths.py`.
- Moved the robot XML, meshes, manifest, and OpenArm license together into
  `resources/openarm/`. Arduino sources are under `firmware/`.
- Moved terminal installation into `scripts/`. The `loop twin`, `loop train`,
  and `loop show` commands and existing model paths are unchanged.

## Terminal shortcuts

- `loop twin`, `loop train`, and `loop show` open the twin, train the student,
  and replay scripted gestures.
- Training uses a new timestamped output path. Both viewers default to the last
  successfully trained model, with `--model` available for an explicit choice.
- A Windows installer registers the command for PowerShell and Command Prompt
  using the repository's virtual environment.

## PC model and standalone Arduino modes

- One UNO R4 sketch now supports PC classifier inference and independent board-
  side tilt rules. Rules run without USB data or a model file; boot selects rules.
- Holding D4 for 1.2 seconds toggles modes once. Short presses retain D2/D3/D4
  targets of 3/5/15 seconds, with debounce and no short press after a hold.
- Both modes flash D9 and the built-in LED for three seconds on a too-short
  gesture interval. The first gesture starts timing without an alert.
- The desktop loads the student model automatically with `--port`, displays the
  current operating mode, and offers a matching mode button. `--control-mode rules`
  starts with board-side detection; arm visualization remains available.
- The board reports counts and results in rules mode; PC inference and outgoing
  alerts stop in that mode. Mode changes reset gesture history. CSV output records
  the operating mode and omits inactive classifier scores.
- A host heartbeat supports automatic fallback to rules after three seconds.
  Closing the application requests rules immediately. Heartbeats do not override
  a physical button selection. USB backpressure cannot block the sensing loop.

## Gesture recognition and student evaluation

PaceLoop now includes a small IMU gesture classifier and a visual replay workflow.
It learns to distinguish scripted hand-to-mouth movements from head scratching,
face touching, and other arm movements. Classification can replace the original
tilt trigger while retaining the existing movement timer and Arduino LED bridge.

### Added

- A NumPy MLP with 208 input features, 32 hidden units, four outputs, and 6,820
  learned parameters. It reads 2.4-second windows of six-axis IMU measurements.
- Scripted OpenArm motion generation and virtual wrist IMU recording, with varied
  motion timing, poses, sensor orientation, bias, and noise.
- A teacher/student data split: student archives contain only timestamps and IMU
  samples; labels, joint trajectories, and wrist positions are stored separately.
- Independent prediction and evaluation commands. The student saves predictions
  before the evaluator loads teacher labels to calculate cross-entropy and event
  metrics.
- Reproducible offline training with Adam, class-weighted cross-entropy, L2
  regularization, validation-based checkpoint selection, and saved epoch history.
- Trained checkpoints, test reports, and loss history under `models/`. The default
  student checkpoint is `gesture_student_v2.npz` (28,830 bytes); the first prototype
  is retained for comparison.
- An English-language gesture viewer with action selection, pause/replay, playback
  speed, new variations, class scores, bite events, IMU traces, and per-window loss.
  Teacher labels can be shown for inspection; they are always excluded from model
  inputs. A separate chart displays training and validation loss.
- An opaque spherical head centered above the OpenArm torso, using the same dark
  gray material. This is a visual guide and does not affect sensor generation.
- `--gesture-demo` and `--gesture-model` application options, plus classifier labels,
  scores, and model provenance in runtime CSV output.

### Training results

The recorded dataset contains 736 simulated episodes across 92 virtual wearing
sessions: 480 training, 112 validation, and 144 test episodes. Sessions are kept
separate across splits. Training stopped after 69 epochs and selected epoch 44.

| Metric | Result |
| --- | ---: |
| Training cross-entropy at selected epoch | 0.0755 |
| Validation cross-entropy at selected epoch | 0.1732 |
| IMU-only test cross-entropy | 0.2017 |
| Test bite events detected | 34 / 36 |
| False bite events | 6 |
| Event precision / recall | 85.0% / 94.4% |

These results describe procedural simulation data. Real eating gestures have not
been validated, and the model does not observe food, utensil contact, or swallowing.
The viewer performs inference with fixed weights; watching a replay does not train
the network. The teacher is the motion script, and the learning method is offline
supervised classification.

### Usage

Start the viewer with `python loop.py show`. For the training pipeline,
model files, loss definitions, and hardware usage, see
[the gesture model guide](docs/gesture-model.md). The code layout is described in
[the development guide](docs/development.md).
