"""Smoke tests for the G1 robot models and the exported policy (run in CI on every push)."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import onnx
import pytest

REPO = Path(__file__).resolve().parents[1]
DESC = REPO / "unitree_g1_description"
MJCF_FILES = ["g1_29dof.xml", "g1.xml", "scene_cath.xml", "scene_cath_mujoco.xml"]
POLICY_ONNX = REPO / "checkpoints" / "mjlab_velocity_flat" / "policy.onnx"


@pytest.mark.parametrize("name", MJCF_FILES)
def test_mjcf_loads_with_29_actuators(name):
    model = mujoco.MjModel.from_xml_path(str(DESC / "urdf_xml" / name))
    assert model.nu == 29
    assert model.nq == 36  # 7 floating base + 29 joints


@pytest.mark.parametrize("name", ["scene_cath.xml", "scene_cath_mujoco.xml"])
def test_scene_steps_without_nan(name):
    model = mujoco.MjModel.from_xml_path(str(DESC / "urdf_xml" / name))
    data = mujoco.MjData(model)
    for _ in range(int(1.0 / model.opt.timestep)):  # one second, zero control
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()


def test_urdf_meshes_exist():
    root = ET.parse(DESC / "urdf_xml" / "g1_29dof.urdf").getroot()
    meshes = {m.attrib["filename"] for m in root.iter("mesh")}
    assert meshes
    for uri in meshes:
        path = DESC / "meshes" / Path(uri.split("unitree_g1_description/")[-1]).name
        assert path.exists(), f"missing mesh {uri}"


def test_urdf_has_29_actuated_joints():
    root = ET.parse(DESC / "urdf_xml" / "g1_29dof.urdf").getroot()
    actuated = [j for j in root.iter("joint") if j.attrib["type"] in ("revolute", "continuous")]
    assert len(actuated) == 29


def test_policy_io_matches_model():
    """The exported policy must agree with the MJCF on joint order (needed for sim-to-sim)."""
    policy = onnx.load(str(POLICY_ONNX))
    dims = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    assert dims(policy.graph.input[0]) == [1, 99]
    assert dims(policy.graph.output[0]) == [1, 29]

    meta = {p.key: p.value for p in policy.metadata_props}
    policy_joints = meta["joint_names"].split(",")
    for name in ["g1.xml", "g1_29dof.xml"]:
        model = mujoco.MjModel.from_xml_path(str(DESC / "urdf_xml" / name))
        actuated = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
        assert actuated == policy_joints, f"joint order differs in {name}"
