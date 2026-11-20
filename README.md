# PaceLoop

Forearm motion tracking and movement pacing with an OpenArm simulation.

PaceLoop reads a wrist-mounted IMU through an Arduino UNO R4 Minima, estimates orientation, and maps forearm movement to the right arm of a MuJoCo model. A desktop interface provides calibration, manual joint control, signal plots, and interval feedback.

The model includes the OpenArm torso and both arms. The left arm is fixed in a lowered pose; the right arm and gripper are actuated. Gripper bands are rigid meshes, and camera geometry is omitted.

## Setup

Python 3.11 is recommended. On Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python run_twin.py
```

The last command starts the simulation with a virtual wrist IMU. Model assets are included, so ROS and a separate OpenArm installation are not required. Python must include Tkinter for the desktop interface.

## Connect an IMU

Upload [PaceLoop_UNO_R4.ino](arduino/PaceLoop_UNO_R4/PaceLoop_UNO_R4.ino) to the Arduino, close the serial monitor, and start:

```powershell
.\.venv\Scripts\python run_twin.py --port COM3
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
| Arduino D2 / D3 / D4 | Select a target interval of 3 / 5 / 15 seconds |

Use `--manual` to read the real IMU with arm following initially disabled. Use `--imu-source sim` with `--port` to retain virtual sensor control while using the Arduino buttons and LED.

Each run saves a CSV under `logs/`. `--csv PATH` selects a new output file; existing files are never overwritten.

## Tracking and model limits

The single-IMU mapping distributes lifting, horizontal swing, and twist across selected robot joints. It does not measure all seven arm joints or wrist position. VQF estimates orientation and gyro bias, but a six-axis sensor has no absolute heading reference. Use **F9** at the known starting pose to correct accumulated heading error.

Movement intervals use an experimental tilt-based gesture detector. The Arduino bridge sends mode and LED feedback; it does not command physical OpenArm motors.

See [model and tracking details](docs/model.md) for joint mappings, motion limits, and asset provenance.

## Development

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Tests cover sensor parsing, calibration, drift handling, pose mapping, trajectory continuity, and headless MuJoCo physics. They do not require connected hardware. [Development notes](docs/development.md) describe the layout and model rebuild process.

Third-party assets retain their original terms. See [third-party notices](THIRD_PARTY_NOTICES.md) and [OpenArm's Apache-2.0 license](OPENARM_LICENSE).
