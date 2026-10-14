"""Run a trained SB3 PPO standing policy in the MuJoCo viewer; the env resets when the robot falls."""

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
    parser.add_argument("--model", default=str(SB3_STAND_MODEL), help="SB3 PPO .zip checkpoint")
    args = parser.parse_args()

    env = G1StandEnv(args.scene)
    model = PPO.load(args.model, env=env, device="cpu")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("Model loaded. Starting balance test...")
        obs, _ = env.reset()

        while viewer.is_running():
            step_start = time.time()

            # Deterministic actions: use the best learned strategy
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()
                print("Robot fell! Resetting environment...")

            with viewer.lock():
                viewer.sync()

            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
