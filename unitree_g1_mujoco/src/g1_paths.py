"""Repository-relative paths, so every script runs from any working directory."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

URDF_XML_DIR = REPO_ROOT / "unitree_g1_description" / "urdf_xml"
# g1_29dof.xml inside the factory-style test scene (machines, tables, goal table)
SCENE_29DOF = URDF_XML_DIR / "scene_cath.xml"
# g1.xml (G1 rev 1.0 model) inside the same test scene
SCENE_REV1 = URDF_XML_DIR / "scene_cath_mujoco.xml"

CHECKPOINTS = REPO_ROOT / "checkpoints"
MJLAB_CHECKPOINT = CHECKPOINTS / "mjlab_velocity_flat" / "model_29999.pt"
MJLAB_ONNX = CHECKPOINTS / "mjlab_velocity_flat" / "policy.onnx"
SB3_STAND_MODEL = CHECKPOINTS / "sb3_ppo_stand" / "g1_self_balance_final.zip"
