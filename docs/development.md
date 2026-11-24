# Development

## Layout

```text
paceloop/app/          Twin and gesture replay interfaces
paceloop/training/     Motion generation, preparation, training, prediction, evaluation
paceloop/learning/     Network definition, IMU features, data loading, metrics
paceloop/bridge/       Serial transport, attitude estimation, forearm mapping
paceloop/simulation/   Robot loading, motion control, model construction
paceloop/cli.py        Command dispatch and trained-model selection
paceloop/paths.py      Shared repository and resource locations
firmware/             UNO R4 firmware
resources/openarm/    Robot XML, packaged meshes, manifest, and license
models/               Trained checkpoints, metrics, and loss history
scripts/              Terminal installation
loop.py / loop.cmd    Stable command entry points
```

Runtime data, local environments, build output, and editor files are ignored by Git.

| What to change | File |
| --- | --- |
| Motion trajectories and simulated sensor noise | `paceloop/training/motion.py` |
| Optimizer, epochs, and training parameters | `paceloop/training/train.py` |
| Network, input features, and event gating | `paceloop/learning/model.py` |
| Serial protocol and attitude estimation | `paceloop/bridge/arduino.py` |
| IMU-to-arm mapping and motion response | `paceloop/bridge/follow.py` |
| Twin controls and timing display | `paceloop/app/twin.py` |
| Gesture replay and loss display | `paceloop/app/viewer.py` |
| Board buttons, standalone rules, and LED timing | `firmware/PaceLoop_UNO_R4/PaceLoopCore.h` |
| Robot joints, bodies, and physics settings | `resources/openarm/arm.xml` |

Run Python tools as modules from the repository, for example
`python -m paceloop.training.motion --help`. Regular use goes through
`loop twin`, `loop train`, and `loop show`. The root launchers stay in place so
installed terminal commands continue to work after source files move.

## Gesture network and training

`paceloop/learning/model.py` defines `TinyGestureModel`: a `208 -> 32 (ReLU) -> 4 (Softmax)`
network. `window_features()` converts each 120-by-6 IMU window into 208 features.
`GestureRecognizer` resamples timestamps and applies the event gate. Runtime model
loading reads the saved `.npz` arrays; it does not initialize random parameters.

`paceloop/training/train.py`, in `train()`, initializes both weight matrices from a
zero-mean normal distribution with standard deviation `sqrt(2 / fan_in)`, sets
biases and Adam moments to zero, and fixes the random seed to 3107. Training uses
manual NumPy backpropagation, batch size 256, Adam learning rate 0.001, and L2
coefficient 0.0001. The selected checkpoint minimizes class-weighted validation
cross-entropy with early stopping after 25 epochs without sufficient improvement.

| File | Role |
| --- | --- |
| `paceloop/training/motion.py` | Script motion and record virtual IMU plus teacher labels |
| `paceloop/training/prepare.py` | Separate sensor-only input files from teacher records |
| `paceloop/training/train.py` | Initialize, train, select, and save the student |
| `paceloop/training/predict.py` | Predict from saved model and IMU-only recordings |
| `paceloop/training/evaluate.py` | Score saved predictions against independently loaded labels |
| `paceloop/app/viewer.py` | Replay motion, inspect predictions, and display loss |

The default checkpoint is `models/gesture_student_v2.npz`. The teacher is the
scripted label source, and the student is an offline supervised classifier. It
does not control the robot or collect trajectories through a learned policy.
See [the gesture guide](gesture-model.md) for runnable commands and reported results.

## Formatting and firmware compilation

Install the development dependencies and run the style checks:

```powershell
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
```

CI runs these checks on Windows. Compile the Arduino firmware with the Renesas
UNO core installed:

```powershell
arduino-cli compile --fqbn arduino:renesas_uno:minima firmware/PaceLoop_UNO_R4
```

`PaceLoopCore.h` contains button handling, mode selection, LED timing, and the
standalone tilt detector. The sketch handles I2C sampling and serial messages.
See [hardware setup](hardware.md) for upload instructions and the protocol.

## Rebuild the model

The committed model is ready to run. Rebuilding requires a Blue_Berry_Harvesting_Bot checkout containing its gripper description, mounting configuration, and OpenArm source assets:

```powershell
python -m paceloop.simulation.build --source ../Blue_Berry_Harvesting_Bot --output ./model-build
```

The builder copies referenced meshes, removes camera geometry, fixes the left-arm pose, and settles the home keyframe. Inspect the generated model before replacing the committed assets. The source checkout is read-only during the build.
