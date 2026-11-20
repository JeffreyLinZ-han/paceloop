# Development

## Layout

```text
run_twin.py         Desktop interface, simulation loop, and gesture timing
imu_hardware.py     Arduino transport and VQF orientation estimation
imu_follow.py       Forearm mapping and continuous target tracking
openarm_model.py    Model loading and manual motion control
arm.xml            MuJoCo model
assets/            Meshes referenced by the model
arduino/           UNO R4 firmware
tests/             Unit tests and headless physics checks
tools/             Model builder
```

Runtime data, local environments, build output, and editor files are ignored by Git.

## Tests

From the repository root:

```powershell
python -m unittest discover -s tests -v
```

The suite checks serial parsing and timestamps, still calibration, known orientation trajectories, bias changes, recentering, target limits, and model behavior. It opens neither a serial port nor a viewer.

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
