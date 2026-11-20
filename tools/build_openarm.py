"""Package the full OpenArm torso and both arms, with only the right arm moving.

Only the output directory is written. The source checkout is read-only.
"""

import argparse
import hashlib
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from openarm_model import compile_model


def build(source, output):
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package = source / "src/blueberry_gripper_description"
    sys.path.insert(0, str(package))
    from blueberry_gripper_description.mjcf import attach_gripper, initialize_gripper

    origin = source / "upstream/openarm_mujoco/v1/openarm_bimanual.xml"
    root = ET.parse(origin).getroot()
    root.set("model", "PaceLoop OpenArm - rigid bands, no cameras")
    compiler = root.find("compiler")
    meshdir = origin.parent / compiler.attrib.pop("meshdir", "")
    compiler.set("angle", "radian")
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str((meshdir / mesh.get("file")).resolve()))
    removed = set()
    for side in ("left", "right"):
        wrist = root.find(f'.//body[@name="openarm_{side}_link7"]')
        stock = wrist.find(f'body[@name="openarm_{side}_hand"]')
        removed.update(e.get("name") for e in stock.iter() if e.get("name"))
        wrist.remove(stock)
    for tag in ("tendon", "equality", "keyframe"):
        section = root.find(tag)
        if section is not None:
            root.remove(section)
    for tag in ("actuator", "contact"):
        for element in list(root.find(tag)):
            if any(value in removed for value in element.attrib.values()):
                root.find(tag).remove(element)

    mounts = json.loads((source / "config/mounts.json").read_text())
    for side in ("left", "right"):
        mount = mounts[side]
        if not np.allclose(mount["rpy"], 0):
            raise ValueError("Recheck the source flange orientation before rebuilding.")
        wrist = root.find(f'.//body[@name="openarm_{side}_link7"]')
        attach_gripper(root, wrist, side + "_bb_", mount["xyz"], band_model="rigid")
    metadata = json.loads((package / "config/gripper.json").read_text())
    camera_geoms = {
        side + "_bb_" + name
        for side in ("left", "right")
        for name in metadata["wrist_camera"]["visual_geometry_names"]
    }
    removed_camera_geoms = []
    for parent in root.iter():
        for element in list(parent):
            if element.tag == "camera" or (
                element.tag == "geom" and element.get("name") in camera_geoms
            ):
                if element.tag == "geom":
                    removed_camera_geoms.append(element.get("name"))
                parent.remove(element)
    if set(removed_camera_geoms) != camera_geoms:
        raise ValueError("Not all camera housing/bracket geometries were found.")
    mesh_files = {mesh.get("name"): mesh.get("file", "") for mesh in root.findall("asset/mesh")}
    for geom in root.findall(".//body/geom"):
        if mesh_files.get(geom.get("mesh"), "").endswith("Part_28.stl"):
            geom.set("rgba", "0.16 0.18 0.20 1")

    wrist = root.find('.//body[@name="openarm_right_link7"]')
    ET.SubElement(wrist, "site", name="imu", pos="0 0 0.075", size="0.009", rgba="0.1 0.9 0.65 1")
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "accelerometer", name="imu_accel", site="imu")
    ET.SubElement(sensors, "gyro", name="imu_gyro", site="imu")
    ET.SubElement(sensors, "framequat", name="imu_quat", objtype="site", objname="imu")
    ET.SubElement(sensors, "framepos", name="wrist_pos", objtype="site", objname="imu")
    root.find("option").set("timestep", "0.002")
    root.find("option").set("integrator", "implicitfast")
    root.find("option").set("iterations", "100")
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "text", name="band_model", data="rigid")
    ET.SubElement(custom, "text", name="paceloop_model", data="openarm_full_body_right_active")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual, "global", azimuth="150", elevation="-15", offwidth="1280", offheight="960"
    )
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7")
    ET.SubElement(root, "statistic", center="0.12 0 0.50", extent="1.25")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", directional="true")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        pos="0 0 -0.02",
        size="2 2 0.05",
        rgba="0.22 0.27 0.32 1",
    )

    used_meshes = {e.get("mesh") for e in root.iter() if e.get("mesh")}
    manifest = []
    for mesh in list(root.find("asset")):
        if mesh.tag != "mesh":
            continue
        if mesh.get("name") not in used_meshes:
            root.find("asset").remove(mesh)
            continue
        original = Path(mesh.get("file"))
        relative = Path("assets") / original.relative_to(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        mesh.set("file", relative.as_posix())
        manifest.append(
            {
                "file": relative.as_posix(),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )

    compiler.set("meshdir", str(output))
    model = compile_model(root, output)
    data = mujoco.MjData(model)
    for side in ("left", "right"):
        data.joint(f"openarm_{side}_joint4").qpos[0] = 0.0 if side == "left" else np.pi / 2
        initialize_gripper(model, data, side + "_bb_")
    data.joint("openarm_left_joint2").qpos[0] = math.radians(-8)
    mujoco.mj_forward(model, data)
    # Bake the full left arm's lowered pose into fixed bodies. This retains
    # all CAD, mass and collision shapes without solving any left-side joints.
    left = root.find('.//body[@name="openarm_left_link0"]')
    fixed_names = {e.get("name") for e in left.iter() if e.get("name")}
    left_world = {}
    for body in left.iter("body"):
        bid = model.body(body.get("name")).id
        pid = model.body_parentid[bid]
        rotation = data.xmat[pid].reshape(3, 3)
        local_pos = rotation.T @ (data.xpos[bid] - data.xpos[pid])
        parent_inverse = data.xquat[pid].copy()
        parent_inverse[1:] *= -1
        local_quat = np.empty(4)
        mujoco.mju_mulQuat(local_quat, parent_inverse, data.xquat[bid])
        body.set("pos", " ".join(format(v, ".17g") for v in local_pos))
        body.set("quat", " ".join(format(v, ".17g") for v in local_quat))
        for joint in list(body.findall("joint")):
            body.remove(joint)
        left_world[body.get("name")] = data.xpos[bid].copy()
    for tag in ("actuator", "equality", "contact"):
        section = root.find(tag)
        for element in list(section):
            if any(value in fixed_names for value in element.attrib.values()):
                section.remove(element)
    model = compile_model(root, output)
    data = mujoco.MjData(model)
    data.joint("openarm_right_joint4").qpos[0] = np.pi / 2
    for j in range(1, 8):
        data.actuator(f"right_joint{j}_ctrl").ctrl[0] = data.joint(f"openarm_right_joint{j}").qpos[
            0
        ]
    initialize_gripper(model, data, "right_bb_")
    mujoco.mj_forward(model, data)
    for name, position in left_world.items():
        np.testing.assert_allclose(data.body(name).xpos, position, atol=1e-10)
    # Start in static equilibrium so gravity does not kick the wrist on launch.
    for _ in range(round(4.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    if np.max(np.abs(data.qvel)) > 1e-3 or any(w.number for w in data.warning):
        raise RuntimeError("Home pose did not settle cleanly.")
    frame = ET.SubElement(ET.SubElement(root, "keyframe"), "key", name="home")
    for field in ("qpos", "qvel", "ctrl", "act"):
        values = getattr(data, field)
        if len(values):
            frame.set(field, " ".join(format(v, ".17g") for v in values))
    compiler.attrib.pop("meshdir")
    root.insert(
        0,
        ET.Comment(
            "OpenArm source: Copyright 2025 Enactic, Inc.; Apache-2.0. See OPENARM_LICENSE."
        ),
    )
    ET.indent(root)
    ET.ElementTree(root).write(output / "arm.xml", encoding="utf-8", xml_declaration=True)
    shutil.copy2(source / "upstream/openarm_mujoco/LICENSE", output / "OPENARM_LICENSE")
    report = {
        "source_project": source.name,
        "source_model": origin.relative_to(source).as_posix(),
        "arm": "complete OpenArm v1 torso and two arms; right arm active, left lowered at side",
        "left_pose_degrees": {"joint2": -8, "joint4": 0},
        "startup": "home keyframe settled under gravity for 4 simulation seconds",
        "imu": "one IMU on openarm_right_link7 only",
        "band": "four rigid Part_28 CAD geometries fixed to finger links; no band joints",
        "camera_geometry_removed": sorted(removed_camera_geoms),
        "dofs": model.nv,
        "actuators": model.nu,
        "cameras": model.ncam,
        "physics_timestep_s": model.opt.timestep,
        "gripper_inertia_note": metadata["inertia_basis"],
        "mount_note": metadata["mount_adapter"],
        "meshes": list({entry["file"]: entry for entry in manifest}.values()),
    }
    (output / "openarm_assets.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "meshes"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    build(args.source, args.output)
