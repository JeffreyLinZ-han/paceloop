# Development

## Layout

```text
run_twin.py         Desktop interface, simulation loop, and gesture timing
imu_hardware.py     Arduino transport and VQF orientation estimation
imu_follow.py       Forearm mapping and continuous target tracking
openarm_model.py    Model loading and manual motion control
gesture_model.py    IMU features, MLP forward pass, weights, streaming recognition
gesture_data.py     Strict IMU-only recording loader
gesture_metrics.py Teacher-side cross-entropy and classification/event metrics
arm.xml            MuJoCo model
assets/            Meshes referenced by the model
arduino/           UNO R4 firmware
models/            Trained checkpoints, metrics, and epoch loss history
tests/             Unit tests and headless physics checks
tools/             Model builder, gesture generation, training, evaluation, viewer
```

Runtime data, local environments, build output, and editor files are ignored by Git.

## Gesture network and training

`gesture_model.py` defines `TinyGestureModel`: a `208 -> 32 (ReLU) -> 4 (Softmax)`
network. `window_features()` converts each 120-by-6 IMU window into 208 features.
`GestureRecognizer` resamples timestamps and applies the event gate. Runtime model
loading reads the saved `.npz` arrays; it does not initialize random parameters.

`tools/train_gestures.py`, in `train()`, initializes both weight matrices from a
zero-mean normal distribution with standard deviation `sqrt(2 / fan_in)`, sets
biases and Adam moments to zero, and fixes the random seed to 3107. Training uses
manual NumPy backpropagation, batch size 256, Adam learning rate 0.001, and L2
coefficient 0.0001. The selected checkpoint minimizes class-weighted validation
cross-entropy with early stopping after 25 epochs without sufficient improvement.

| File | Role |
| --- | --- |
| `tools/simulate_gestures.py` | Script motion and record virtual IMU plus teacher labels |
| `tools/prepare_student_data.py` | Separate sensor-only input files from teacher records |
| `tools/train_gestures.py` | Initialize, train, select, and save the student |
| `tools/predict_gestures.py` | Predict from saved model and IMU-only recordings |
| `tools/evaluate_gestures.py` | Score saved predictions against independently loaded labels |
| `tools/gesture_viewer.py` | Replay motion, inspect predictions, and display loss |

The default checkpoint is `models/gesture_student_v2.npz`. The teacher is the
scripted label source, and the student is an offline supervised classifier. It
does not control the robot or collect trajectories through a learned policy.
See [the gesture guide](gesture-model.md) for runnable commands and reported results.

## Tests

From the repository root:

```powershell
python -m unittest discover -s tests -v
```

The suite checks serial parsing and timestamps, still calibration, known orientation trajectories, bias changes, recentering, target limits, model behavior, gesture inference, and student input isolation. It opens neither a serial port nor a viewer.

Install `requirements-dev.txt` to run the same style checks as CI:

```powershell
python -m ruff check .
python -m ruff format --check .
```

`tests/data/forearm_imu.csv` contains 1,605 recorded hardware frames for a replay test. Replay is an offline test input, not an application fallback for disconnected hardware. Known synthetic trajectories provide independent ground truth for orientation-error tests.

## Rebuild the model

The committed model is ready to run. Rebuilding requires a Blue_Berry_Harvesting_Bot checkout containing its gripper description, mounting configuration, and OpenArm source assets:

```powershell
python -m tools.build_openarm --source ../Blue_Berry_Harvesting_Bot --output ./model-build
```

The builder copies referenced meshes, removes camera geometry, fixes the left-arm pose, and settles the home keyframe. Inspect the generated model before replacing the committed assets. The source checkout is read-only during the build.
