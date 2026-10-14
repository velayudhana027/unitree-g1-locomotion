"""Open the G1 model in the native MuJoCo viewer to inspect geometry, joint axes and frames."""

import argparse
import time

import mujoco
import mujoco.viewer

from g1_paths import SCENE_29DOF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=str(SCENE_29DOF), help="MJCF scene to load")
    args = parser.parse_args()

    # MuJoCo resolves meshdir="../meshes/" relative to the model file
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    print(f"Loaded {args.scene}: {model.nu} actuators, {model.nbody} bodies")

    # The passive viewer runs in a separate thread while this loop steps the physics
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            # Keep real-time speed
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
