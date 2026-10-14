"""Play the trained mjlab velocity policy (Mjlab-Velocity-Flat-Unitree-G1) in the Viser web viewer.

Needs mjlab installed (tested with mjlab 1.2.0). After starting, open the URL
printed in the terminal (usually http://localhost:8080) and set velocity
commands from the "Commands" panel.
"""

import argparse
from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import ViserPlayViewer

from g1_paths import MJLAB_CHECKPOINT

TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(MJLAB_CHECKPOINT), help="mjlab .pt checkpoint")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    # Environment: a single instance for viewing
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load the trained policy
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True})
    policy = runner.get_inference_policy(device=args.device)

    print("Starting the Viser web viewer; open the URL it prints (usually http://localhost:8080).")
    ViserPlayViewer(env, policy).run()


if __name__ == "__main__":
    main()
