"""Watch an intermediate SB3 PPO checkpoint while training is still running."""

import argparse
import time

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from g1_paths import SB3_STAND_MODEL, SCENE_29DOF
from train_g1_balance import G1StandEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=str(SCENE_29DOF), help="MJCF scene to load")
    parser.add_argument(
        "--model",
        default=str(SB3_STAND_MODEL),
        help="checkpoint to watch, e.g. ./models/g1_checkpoints/g1_stand_model_900000_steps.zip",
    )
    args = parser.parse_args()

    env = G1StandEnv(args.scene)
    model = PPO.load(args.model, device="cpu")

    obs, _ = env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()

            viewer.sync()
            time.sleep(env.model.opt.timestep)


if __name__ == "__main__":
    main()
