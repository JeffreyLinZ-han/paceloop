# PaceLoop

Forearm motion tracking and movement pacing with an OpenArm simulation.

PaceLoop reads a wrist-mounted IMU through an Arduino UNO R4 Minima, estimates orientation, and maps forearm movement to the right arm of a MuJoCo model. A desktop interface provides calibration, manual joint control, signal plots, and interval feedback.

The model includes the OpenArm torso and both arms. The left arm is fixed in a lowered pose; the right arm and gripper are actuated. Gripper bands are rigid meshes, and camera geometry is omitted.

See the [change summary](CHANGELOG.md) for the new gesture classifier, student training
pipeline, interactive viewer, and evaluation results.

## Repository layout

```text
paceloop/                 Editable Python source
  app/                    Twin and gesture viewer
  training/               Scripted motions, data preparation, training, evaluation
  learning/               Network, IMU features, data loading, loss metrics
  bridge/                 Arduino serial connection and IMU-to-arm mapping
  simulation/             OpenArm loading, joint control, and model builder
  cli.py                  loop command handling
  paths.py                Shared repository and robot resource paths
firmware/PaceLoop_UNO_R4/  Arduino sketch and standalone rules
resources/openarm/        Packaged robot XML, meshes, asset manifest, and license
models/                   Trained weights, training history, and reports
scripts/                  Terminal command installation
docs/                     Hardware, training, and development guides
loop.py / loop.cmd        Command entry points
```

For motion changes, edit `paceloop/training/motion.py`; for training settings,
edit `paceloop/training/train.py`; for network changes, edit
`paceloop/learning/model.py`. Hardware connection and tracking code lives in
`paceloop/bridge/`. The packaged meshes under `resources/openarm/assets/`
normally stay unchanged. Runtime datasets, logs, and build output are kept in
`datasets/`, `logs/`, and `build/`, all excluded from Git.

## Setup

Python 3.11 is recommended. On Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\scripts\install_loop.ps1
```

Open a new PowerShell terminal after installation. The `loop` command uses this
checkout's virtual environment, from any working directory. Model assets are
included, so ROS and a separate OpenArm installation are not required. Python
must include Tkinter for the desktop interface.

## Terminal commands

| Command | Action |
| --- | --- |
| `loop twin` | Open the OpenArm twin with a virtual IMU and gesture classifier |
| `loop twin --port COM3` | Follow the real IMU and run PC model detection |
| `loop twin --port COM3 --control-mode rules` | Monitor Arduino rules and visualize the arm |
| `loop train` | Train from the local `datasets/gesture_student_v2` recordings |
| `loop train --epochs 50` | Train for up to 50 epochs |
| `loop show` | Replay scripted gestures with model predictions |
| `loop show --show-loss` | Open the replay and saved training loss curves |
| `loop --help` | Show the command list |

Training creates a new timestamped model under `models/runs/`. After a successful
run, `loop twin` and `loop show` select that model automatically. Until then, they
use the bundled student checkpoint. `--model PATH.npz` selects a particular model
for either viewer. Explicit relative paths are resolved from the terminal's
working directory.

Press **Ctrl+C** to stop a terminal command, or close the viewer to exit.
Training saves weights after the training loop completes; interrupting the loop
does not save a checkpoint or change the model selected by the shortcuts.
Training data is local and excluded from Git; see [data generation](docs/gesture-model.md#training-and-evaluation)
if it is missing from a fresh checkout.

The installer adds a user-level command and a PowerShell function. Restart the
terminal application to refresh PATH for Command Prompt. Without installation,
use `.\loop.cmd twin` or `.\.venv\Scripts\python loop.py twin` from the repository.

## Connect an IMU

Upload [PaceLoop_UNO_R4.ino](firmware/PaceLoop_UNO_R4/PaceLoop_UNO_R4.ino) to the Arduino, close the serial monitor, and start:

```powershell
loop twin --port COM3
```

Replace `COM3` with the board's serial port. See [hardware setup](docs/hardware.md) for wiring, firmware upload, and the serial protocol.

1. Secure the sensor to the forearm or wrist. Bend the elbow about 90 degrees with the forearm level and pointing forward.
2. Hold still for about two seconds to calibrate.
3. When prompted, raise the forearm at least 12 degrees without twisting it. This teaches the lifting direction.
4. Once **SYNC LIVE** appears, lower back to the starting pose and move normally.

| Control | Action |
| --- | --- |
| Follow forearm IMU | Toggle between IMU following and manual joint sliders |
| Calibrate neutral | Repeat still calibration and learn a new lifting direction |
| Recenter to start / F9 | After returning to the original physical pose, correct the mapping reference |
| Gripper | Open or close the simulated gripper independently |
| Arduino D2 / D3 / D4, short press | Select a target interval of 3 / 5 / 15 seconds |
| Arduino D4, hold for 1.2 seconds | Switch between PC model and Arduino rules |
| Switch mode | Change the same operating mode from the desktop interface |

Use `--manual` to read the real IMU with arm following initially disabled. Use `--imu-source sim` with `--port` to retain virtual sensor control while using the Arduino buttons and LED.

Each run saves a CSV under `logs/`. `--csv PATH` selects a new output file; existing files are never overwritten.

## Two operating modes

Upload the v3 sketch once to enable both modes and the D4 switch.

| Mode | Detection runs on | Requires a computer? |
| --- | --- | --- |
| **PC model** | Python, using `models/gesture_student_v2.npz` | Yes: run PaceLoop with `--port COM3` |
| **Arduino rules** | UNO R4, using calibrated tilt and movement thresholds | No: power the board and IMU |

The board starts in **Arduino rules**. Hold the sensor still in your neutral
forearm pose for about two seconds, then lift and return between gestures.
Starting the desktop application selects **PC model** and loads the bundled model.
To start the desktop as a monitor of board-side detection instead:

```powershell
loop twin --port COM3 --control-mode rules
```

**Hold D4 for 1.2 seconds to switch modes.** Releasing a long press does not change
the target interval; a short D4 press still selects 15 seconds. Switching resets
gesture counts and interval history. Entering rules mode also repeats still
calibration. The desktop displays the active mode and keeps the arm visualization
available in either mode when connected.

Both modes flash **D9 and the built-in LED** for three seconds when the interval
between detected gestures is shorter than the selected target. The first gesture
starts timing without an alert. A mode change gives one slow pulse for rules or
two for PC; an active overspeed warning takes priority. The boot target is four
seconds until a pace button is pressed.

PC mode requires the application to keep running. If it stops responding for
three seconds, the powered board falls back to rules and recalibrates. Selecting
PC without a running application also returns to rules after three seconds.
Closing the application requests rules immediately. Standalone operation needs
power, but does not need USB data, a serial reader, Python, or a model file.

Rules are a simple tilt proxy and can confuse scratching or face touching with
eating. The PC model attempts to distinguish these gestures, but its training and
evaluation are still simulation-only.

## Gesture recognition

An optional small classifier distinguishes simulated hand-to-mouth gestures from
head scratching, face touching, and other movements. Open the interactive replay:

```powershell
loop show
```

The viewer shows the OpenArm motion, scripted label, model scores, and detected
events. With `--port`, the normal application loads the student model automatically.
Use `--gesture-model PATH.npz` to choose another checkpoint, or to enable learned
detection in a virtual-IMU run. This model is trained on simulation only; see
[training, test results, and limitations](docs/gesture-model.md).

The viewer also displays per-window cross-entropy and provides a **Training loss**
button for the saved training/validation curves. Student inputs contain only IMU;
teacher labels are kept separately for supervision and evaluation. The network is
an offline-trained NumPy MLP. See the [code layout](docs/development.md) to locate its
definition, initialization, and training loop.

## Tracking and model limits

The single-IMU mapping distributes lifting, horizontal swing, and twist across selected robot joints. It does not measure all seven arm joints or wrist position. VQF estimates orientation and gyro bias, but a six-axis sensor has no absolute heading reference. Use **F9** at the known starting pose to correct accumulated heading error.

Without a port or `--gesture-model`, the virtual-IMU run uses the original experimental tilt-based detector. The Arduino bridge sends mode and LED feedback; it does not command physical OpenArm motors.

See [model and tracking details](docs/model.md) for joint mappings, motion limits, and asset provenance.

## Development

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m ruff format --check .
```

CI checks Python style and formatting. [Development notes](docs/development.md)
describe the code layout, firmware compilation, and model rebuild process.

Third-party assets retain their original terms. See [third-party notices](THIRD_PARTY_NOTICES.md) and [OpenArm's Apache-2.0 license](resources/openarm/OPENARM_LICENSE).
