"""Physics, event-timing and portable-asset checks; does not access serial hardware."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from openarm_model import SmoothControls, bite_angle, controls_for, load_model
from run_twin import MODE_NAMES, BiteTimer


class ModelTests(unittest.TestCase):
    def test_model(self):
        root = Path(__file__).resolve().parents[1]
        model, data = load_model(root / "arm.xml")
        assert model.ncam == 0
        assert model.nv == 13 and model.nu == 8
        xml = ET.parse(root / "arm.xml").getroot()
        joint_names = [j.get("name", "") for j in xml.findall(".//body/joint")]
        assert not any("band" in name or "roller" in name for name in joint_names)
        assets = {a.get("name"): a.get("file", "") for a in xml.findall("asset/mesh")}
        bands = [
            g
            for g in xml.findall(".//body/geom")
            if assets.get(g.get("mesh"), "").endswith("/Part_28.stl")
        ]
        assert len(bands) == 4, f"Expected four original rigid CAD bands, found {len(bands)}"
        assert xml.find('.//body[@name="openarm_body_link0"]') is not None
        assert xml.find('.//body[@name="openarm_left_link0"]') is not None
        assert not xml.findall('.//body[@name="openarm_left_link0"]//joint')
        assert xml.find('.//body[@name="openarm_right_link7"]/site[@name="imu"]') is not None
        assert len(xml.findall('.//site[@name="imu"]')) == 1
        left_position = data.body("openarm_left_link7").xpos.copy()
        for path in assets.values():
            assert not Path(path).is_absolute() and (root / path).is_file()
        assert MODE_NAMES[5000] == "MEDIUM" and MODE_NAMES[15000] == "SLOW"

        specs = controls_for(model)
        motion = SmoothControls(model, data, specs)
        neutral = [s.home for s in specs]
        timer = BiteTimer()
        events = []
        max_contact = 0
        max_closure = 0.0
        max_speed = 0.0
        # Three upward elbow excursions with full returns: one too-fast and one OK.
        schedule = [
            (3, 90, 0),
            (5, 130, 0),
            (7, 90, 100),
            (9, 130, 100),
            (13, 90, 0),
            (15, 130, 0),
            (18, 90, 0),
        ]
        stages = []
        for until, j4, gripper in schedule:
            values = neutral.copy()
            values[3], values[-1] = j4, gripper
            while data.time < until:
                motion.set_targets(values)
                motion.step(model.opt.timestep)
                mujoco.mj_step(model, data)
                assert np.isfinite(data.qpos).all() and np.isfinite(data.sensordata).all()
                result = timer.step(data.time, bite_angle(data), data.sensor("imu_gyro").data, 5.0)
                if result:
                    events.append(
                        {
                            "time": float(data.time),
                            "result": result,
                            "interval": timer.previous_interval,
                        }
                    )
                max_contact = max(max_contact, data.ncon)
                max_speed = max(max_speed, float(np.max(np.abs(data.qvel))))
                eq = data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
                if np.any(eq):
                    max_closure = max(
                        max_closure, float(np.max(np.abs(data.efc_pos[: data.nefc][eq])))
                    )
            stages.append(
                {
                    "time": float(data.time),
                    "lift_deg": bite_angle(data),
                    "jaw_rad": float(data.joint("right_bb_HR11J").qpos[0]),
                    "wrist_pos": data.sensor("wrist_pos").data.tolist(),
                }
            )
        np.testing.assert_allclose(data.body("openarm_left_link7").xpos, left_position, atol=1e-12)
        assert left_position[2] < 0.30 and abs(left_position[0]) < 0.08, (
            "Left arm must hang at the side"
        )
        left_tip = data.site("left_bb_gripper_tcp").xpos
        assert left_tip[2] < 0.12 and left_tip[1] > 0.20, (
            "Left gripper must clear the front workspace"
        )
        warnings = {
            mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number
        }
        assert not warnings, warnings
        assert [event["result"] for event in events] == ["FIRST BITE", "TOO FAST", "OK"], events
        assert abs(stages[0]["lift_deg"]) < 10, "Rest pose must rearm detector"
        assert stages[1]["wrist_pos"][2] > stages[0]["wrist_pos"][2] + 0.1, (
            "Bite test must lift the wrist"
        )
        assert stages[3]["jaw_rad"] > 1.1, "Gripper must close under its actuator"
        assert stages[-1]["jaw_rad"] < 0.2, "Gripper must reopen"
        assert max_closure < 0.02, "Gripper linkage drift"
