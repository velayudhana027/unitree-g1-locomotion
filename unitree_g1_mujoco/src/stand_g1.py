"""PD posture-hold baseline for the G1 in MuJoCo.

Joint-space PD control towards a slightly crouched pose, with the
proportional gain ramped up over the first two seconds to avoid snapping.
There is no active balance, so the stance is fragile; this baseline is what
motivated moving to reinforcement learning.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from g1_paths import SCENE_29DOF

KP_FINAL = 350.0
KV = 50.0
RAMP_SECONDS = 2.0


def target_pose(nu):
    """Crouched pose: bent knees and hips, ankles pitched to compensate."""
    pose = np.zeros(nu)
    pose[0], pose[3], pose[4] = -0.4, 0.8, -0.4    # left hip pitch, knee, ankle pitch
    pose[6], pose[9], pose[10] = -0.4, 0.8, -0.4   # right hip pitch, knee, ankle pitch
    return pose


def pd_torques(data, target, kp, kv):
    """tau = kp * (q_target - q) - kv * qdot, skipping the floating base."""
    return kp * (target - data.qpos[7:]) - kv * data.qvel[6:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=str(SCENE_29DOF), help="MJCF scene to load")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    target = target_pose(model.nu)

    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.76  # start close to standing height to limit the impact bounce

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("G1 stand phase: stabilising...")
        while viewer.is_running():
            step_start = time.time()

            # Ramp kp in simulation time
            kp = KP_FINAL * min(1.0, data.time / RAMP_SECONDS)
            data.ctrl[:] = pd_torques(data, target, kp, KV)

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
