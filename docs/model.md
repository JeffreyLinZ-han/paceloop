# Model and tracking

## OpenArm model

`resources/openarm/arm.xml` contains the OpenArm v1 torso and both arms. Only the right arm and right gripper are actuated. The left arm is fixed at the body's side with shoulder J2 at -8 degrees and elbow J4 straight.

The model has 13 scalar degrees of freedom: seven right-arm joints and six gripper linkage joints. It has eight actuators, a 2 ms timestep, and no cameras. One virtual IMU is attached to the right wrist.

Four CAD bands are fixed to the finger links. They retain their mesh geometry but have no independent motion or elastic deformation. The jaw linkages remain articulated. Wrist camera housings and brackets are removed.

Arm inertias and servo limits come from OpenArm. Gripper masses, inertias, and mounting transforms are simulation assumptions inherited from the gripper description. The home keyframe is settled under gravity before use.

## Forearm mapping

Initial still calibration estimates gyro offset and neutral orientation. A deliberate lifting motion identifies the sensor's mounting direction. The forearm should be level during neutral calibration.

| Motion relative to neutral | Robot target |
| --- | --- |
| Lift up to 40 degrees | J4 elbow flexion |
| Additional lift | J1 shoulder elevation |
| Horizontal swing | J3 rotation |
| Forearm twist | J5 rotation |

J2, J6, and J7 retain their starting targets. The gripper remains under manual control.

Mapping uses a 0.4-degree deadband and 35 ms smoothing. Lift is limited to -70/+80 degrees, swing to +/-60, and twist to +/-75; robot joint limits also apply. Heading is held when the forearm is nearly vertical.

Following uses a continuous target tracker with command limits of 150 degrees/s and 900 degrees/s^2. Manual controls use quintic trajectories limited to 90 degrees/s, 360 degrees/s^2, and 2400 degrees/s^3. Actual joint response depends on servo dynamics and load.

## Orientation and recentering

[VQF](https://vqf.readthedocs.io/en/stable/ref_cython.html) performs six-axis orientation fusion and residual gyro-bias estimation. Initial calibration removes the module's fixed offset. The filter uses device timestamps, with interpolation onto a uniform sampling clock.

Acceleration constrains tilt but provides no absolute heading. Bias tracking reduces accumulation; it cannot guarantee a heading reference after arbitrary movement. Press **F9** only after returning to the known starting pose. This resets the mapping reference while retaining the original robot anchor, taught lifting direction, and gyro-bias estimate.

## Data and gesture timing

Hardware CSV rows contain each received sample once. Raw acceleration and gyro values are unchanged. Quaternion columns contain orientation estimates; physical position columns are blank. Joint command and `j*_actual_deg` columns refer to the simulated arm. Bias, rest state, and recenter count are also recorded.

The gesture detector requires more than 25 degrees of neutral-relative tilt and angular speed above 0.15 rad/s. Returning below 10 degrees rearms detection; a 1.2-second minimum gap rejects closely repeated triggers. It is an experimental movement proxy, not a validated eating classifier.

## Asset sources

OpenArm geometry comes from [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco), pinned in the source project to revision `06b4c8ad6d547c3faaa3bf780629ee3e1ac2b0d0`. Gripper geometry comes from the `blueberry_gripper_description` package in the Blue_Berry_Harvesting_Bot project.

`resources/openarm/openarm_assets.json` records relative asset paths and SHA-256 hashes. All meshes required at runtime are included. See [third-party notices](../THIRD_PARTY_NOTICES.md) for applicable terms.
