# Gesture classifier

The optional classifier distinguishes a scripted hand-to-mouth gesture from head
scratching, face touching, and other arm movements. It is a **simulation-trained
prototype**. It has not been validated on recordings of human eating.

## View the model in action

From the repository directory:

```powershell
.\.venv\Scripts\python loop.py show
```

The OpenArm viewer and a control panel open together. Select an action, pause with
Space, replay it, change the movement variant, or cycle through all four actions.
The panel shows the scripted label separately from the model prediction, four
class scores, detected bite events, and the six IMU traces. Playback speed changes
the animation speed while preserving the original sensor timestamps.

The opaque spherical head is a visual guide centered directly above the torso. It
has no contact dynamics and is never a classifier input. The replay uses prescribed
joint trajectories, rather than a physical human or a robot eating real food.
The viewer runs inference with fixed model weights. Training is a separate command;
watching or changing a replay does not update the model.

The student always receives only IMU measurements. **Show teacher labels** controls
their visibility in the panel; it does not change the student's inputs. The displayed
cross-entropy is computed after each prediction, using the corresponding teacher
label. Its mean covers this replay only, not the complete held-out test set.

**Training loss** opens the saved train/validation loss curves. Hover over the chart
to read an epoch. To open both the replay and the curves at startup:

```powershell
.\.venv\Scripts\python -m paceloop.app.viewer --show-loss
```

## Use a real IMU

```powershell
loop twin --port COM3
```

Calibration and arm following work as before. The learned detector
replaces the tilt threshold for bite timing in PC mode.
Detected events use the existing interval timer and Arduino LED bridge. The CSV
adds `gesture_label`, `bite_model_score`, and `gesture_model_source`.

Use `--control-mode rules` for Arduino-side tilt detection, or hold D4 to switch.
`loop twin --model PATH.npz` chooses a checkpoint for real or virtual IMU input.
Scores are model outputs, not calibrated probabilities of eating.

## Inputs and model size

- Six channels: acceleration including gravity in m/s² and angular velocity in rad/s.
- Device timestamps resampled to 50 Hz; 120-sample trailing window (2.4 seconds).
- A prediction every 10 samples (0.2 seconds), after the first complete window.
- 208 features preserving time order, including pooled channels and magnitudes.
- A 32-unit hidden layer and four outputs: 6,820 learned parameters.
- The compressed model is 28,830 bytes. Inference and training use NumPy; no GPU or
  additional machine-learning runtime is required.

An event requires two consecutive high bite scores. A low-score release is needed
before another event, with a minimum 1.2-second event interval. Gaps longer than
0.25 seconds, a sensor restart, or invalid input clear the recognition window.

## Training and evaluation

With the local recordings available, run `loop train`. Use `loop train --epochs 50`
to limit training, or `--output PATH.npz` for a specific new filename. The default
output is a timestamped file under `models/runs/`; a successful run becomes the
default model for `loop twin` and `loop show`.

Press Ctrl+C to stop. The training loop keeps weights in memory until it finishes;
an interruption does not create a resumable checkpoint.

For a fresh checkout, generate and separate the recordings first:

```powershell
.\.venv\Scripts\python -m paceloop.training.motion --output datasets/gestures_v1
.\.venv\Scripts\python -m paceloop.training.prepare --source datasets/gestures_v1 --output datasets/gesture_student_v2
loop train
```

Choose new output directories when regenerating data. Training chooses a new
model filename automatically. Generated recordings under `datasets/` are excluded
from Git.

The teacher is the motion script and its labels; there is no separately trained
teacher network. The student learns supervised classification from the recorded
six-axis IMU. The current motion templates are unchanged from the first prototype.

`prepare_student_data` separates the existing recordings into:

| Directory | Contents | Who uses it |
| --- | --- | --- |
| `imu/train`, `imu/validation`, `imu/test` | Only `time` and six-column `signal` | Student |
| `teacher/train`, `teacher/validation`, `teacher/test` | Labels, scripted action, joints, wrist position and event intervals | Training supervision / evaluation / replay |

Filenames are opaque sensor-content identifiers, with no action names. The student
loader rejects archives containing extra fields such as `labels`, `joints`, or
`action`. Training uses teacher labels to calculate gradients, but the forward pass
always uses only IMU-derived features. Normalization and class weights come from
the training split.

## Blind test and loss

Run prediction and scoring as separate commands with fresh output paths:

```powershell
.\.venv\Scripts\python -m paceloop.training.predict --model models/gesture_student_v2.npz --input datasets/gesture_student_v2/imu/test --output logs/student_predictions
.\.venv\Scripts\python -m paceloop.training.evaluate --predictions logs/student_predictions --teacher datasets/gesture_student_v2/teacher/test --output logs/student_test_loss.json
```

The first command has no teacher-data argument and saves only predictions, timestamps,
and input checksums. The evaluator then loads the saved predictions and the matching
teacher labels. It checks both input hashes and timestamps. It does not load or update
the model.

For four classes, the reported loss is `mean(-log(p_correct_class))`. A uniform
prediction has loss `log(4) = 1.3863`; confident correct predictions approach zero.
The learning objective uses class-weighted cross-entropy plus L2 regularization;
the viewer curves show unweighted cross-entropy for train/validation comparison.
Checkpoint selection uses class-weighted validation cross-entropy. The test loss
is computed only after checkpoint and event-threshold selection.

The second run completed 69 epochs, with epoch 44 selected by validation loss:

| Split | Unweighted cross-entropy |
| --- | ---: |
| Train, selected checkpoint | 0.0755 |
| Validation, selected checkpoint | 0.1732 |
| Blind test | 0.2017 |

See [epoch history](../models/gesture_student_v2.history.csv) and the independent
[blind-test report](../models/gesture_student_v2.blind_test.json). The wider train/test
loss gap is a reason to evaluate more varied motions; these losses do not establish
accuracy on a person wearing the real sensor.

## Simulation and held-out results

The generator evaluates the existing OpenArm model using MuJoCo inverse dynamics
at prescribed joint positions, velocities and accelerations. It reads the wrist
accelerometer and gyro in their local frame; see the
[MuJoCo sensor reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-accelerometer).
Accelerometer readings include gravity. Joint angles, wrist position, action labels,
and head distance are excluded from the network input.

The 736 episodes cover 92 independently seeded virtual wearing sessions. Each
session has varied sensor orientation (up to 25 degrees), biases and noise. Motion
endpoints, duration, holding time and scratching motion also vary. These variations
approximate a limited range of motion; they are not a model of human biomechanics.

Entire sessions are assigned to training (480 episodes), validation (112), or
test (144). Normalization is fitted only to training data; early stopping and the
event threshold use only validation data. The test sessions are evaluated after
those decisions. Adjacent windows from an episode cannot cross split boundaries.

On the held-out synthetic test episodes:

| Event metric | Result |
| --- | ---: |
| Scripted bite events detected | 34 / 36 |
| Missed bite events | 2 |
| False bite events | 6 |
| Event precision | 85.0% |
| Event recall | 94.4% |
| Non-bite episodes with a false event | 6 / 108 |

The complete confusion matrix, split IDs, model checksum and training settings are
in [the metrics file](../models/gesture_student_v2.metrics.json). These are simulated
motion results, not a real-world accuracy estimate. The test generator shares the
same procedural action families as training, with independent parameters/seeds.

Before relying on real alerts, record and label human eating, scratching, face
touching, drinking, and ordinary reaching across several separate wearing sessions.
Drinking is not an explicit training class in this version. A single wrist IMU
cannot distinguish intentions when the wrist trajectories are effectively identical;
it also cannot verify the presence of food or swallowing.
