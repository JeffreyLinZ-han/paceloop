# Robot resources

`openarm/` contains the packaged OpenArm model and gripper geometry:

- `arm.xml`: MuJoCo bodies, joints, actuators, materials, and sensors.
- `assets/`: meshes referenced by the XML; usually left unchanged.
- `openarm_assets.json`: source information and mesh hashes.
- `OPENARM_LICENSE`: the upstream OpenArm license.

Keep this directory structure intact: mesh paths are relative to `arm.xml`.
Application code loads it through `paceloop/paths.py`.

To rebuild the robot from upstream sources, use `paceloop/simulation/build.py`
with a separate output directory. See [development](../docs/development.md) and
[third-party notices](../THIRD_PARTY_NOTICES.md).
