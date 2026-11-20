# Third-party notices

## OpenArm

The OpenArm model and associated meshes are derived from [enactic/openarm_mujoco](https://github.com/enactic/openarm_mujoco).

Copyright 2025 Enactic, Inc. Licensed under Apache-2.0. The complete license is included in [OPENARM_LICENSE](OPENARM_LICENSE).

PaceLoop modifies the model by replacing the stock grippers, fixing the left arm in a lowered pose, removing camera geometry, adding a right-wrist IMU, and settling the home keyframe. These modifications do not change the upstream license.

## Blueberry gripper

Gripper meshes and mounting definitions originate from `blueberry_gripper_description` in Blue_Berry_Harvesting_Bot. The source package declares a **Proprietary** license. These assets retain that designation and are not covered by OpenArm's Apache-2.0 license.

PaceLoop uses rigid band geometry and articulated jaw linkages, with the wrist camera assembly omitted.

## Software dependencies

MuJoCo, NumPy, pySerial, and VQF are installed as dependencies and retain their respective licenses. Their source code is not vendored in this repository.

This repository does not apply a project-wide open-source license to the application or proprietary gripper assets.
