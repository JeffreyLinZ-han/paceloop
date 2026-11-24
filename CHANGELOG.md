# Changelog

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
  metrics. Tests verify that changing teacher labels cannot change predictions.
- Reproducible offline training with Adam, class-weighted cross-entropy, L2
  regularization, validation-based checkpoint selection, and saved epoch history.
- Trained checkpoints, test reports, and loss history under `models/`. The default
  student checkpoint is `gesture_student_v2.npz` (28,830 bytes); the first prototype
  is retained for comparison and regression checks.
- An English-language gesture viewer with action selection, pause/replay, playback
  speed, new variations, class scores, bite events, IMU traces, and per-window loss.
  Teacher labels can be shown for inspection; they are always excluded from model
  inputs. A separate chart displays training and validation loss.
- An opaque spherical head centered above the OpenArm torso, using the same dark
  gray material. This is a visual guide and does not affect sensor generation.
- `--gesture-demo` and `--gesture-model` application options, plus classifier labels,
  scores, and model provenance in runtime CSV output.
- Tests for inference timing, event gating, simulated acceleration, model loading,
  hardware integration, input isolation, target alignment, and loss calculation.

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

### Verification and usage

The local suite passes all 29 tests. Ruff lint and formatting checks pass, and an
independent prediction/evaluation run reproduces the saved test loss without
changing the model checksum. The viewer and training-loss chart were checked on
Windows. Generated recordings, per-episode predictions, logs, screenshots, and
local environments are excluded from Git.

Start the viewer with `python run_twin.py --gesture-demo`. For the training pipeline,
model files, loss definitions, and hardware usage, see
[the gesture model guide](docs/gesture-model.md). The code layout is described in
[the development guide](docs/development.md).
