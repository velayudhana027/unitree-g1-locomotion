"""Gymnasium environment and PPO training script for G1 standing balance (Stable-Baselines3)."""

import argparse
import os

import mujoco
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from g1_paths import SCENE_29DOF

class G1StandEnv(gym.Env):
    def __init__(self, xml_path):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(29,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(69,), dtype=np.float32)
        
        # New: Track last action for smoothing
        self.last_action = np.zeros(29)

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[2:].flatten(), 
            self.data.qvel.flatten()
        ]).astype(np.float32)

    def step(self, action):
        # Apply scaled torque
        self.data.ctrl[:] = action * 50.0 
        mujoco.mj_step(self.model, self.data)
        
        # --- ENHANCED REWARD FUNCTION ---
        height = self.data.qpos[2]
        uprightness = self.data.xmat[self.model.body('torso_link').id, 8] 
        
        # 1. Survival & Posture (Carrot)
        reward = (height * 10.0) + (uprightness * 5.0)
        
        # 2. Anti-Sway: Penalize horizontal velocity of the pelvis (Stops the "Drunk" walk)
        # qvel[:2] are the linear X and Y velocities of the base
        base_vel_xy = self.data.qvel[:2]
        reward -= 2.0 * np.sum(np.square(base_vel_xy))
        
        # 3. Action Smoothing: Penalize large changes between consecutive actions
        # This prevents high-frequency jitter
        reward -= 0.1 * np.sum(np.square(action - self.last_action))
        
        # 4. Energy Efficiency
        reward -= 0.001 * np.sum(np.square(self.data.qvel)) # Penalize general jitter
        reward -= 0.01 * np.sum(np.square(action))         # Penalize high effort

        # Update last action
        self.last_action = action.copy()

        # Termination
        terminated = bool(height < 0.45 or height > 1.0)
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.76 
        
        # Increased noise: Force the robot to handle more difficult starting tilts
        self.data.qpos[7:] += np.random.uniform(-0.1, 0.1, size=29)
        
        # Reset tracking variables
        self.last_action = np.zeros(29)
        return self._get_obs(), {}

def main():
    parser = argparse.ArgumentParser(description="Train a PPO standing policy for the G1 (Stable-Baselines3).")
    parser.add_argument("--scene", default=str(SCENE_29DOF), help="MJCF scene to load")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="total environment steps")
    parser.add_argument("--log-dir", default="./logs/g1_tensorboard/")
    parser.add_argument("--model-dir", default="./models/g1_checkpoints/")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    env = G1StandEnv(args.scene)

    # Save a checkpoint every 100,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=args.model_dir,
        name_prefix="g1_stand_model",
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        tensorboard_log=args.log_dir,
        learning_rate=3e-4,
        batch_size=128,
        n_steps=2048,
    )

    print(f"Starting training. Monitor with: tensorboard --logdir {args.log_dir}")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True,  # needs: pip install tqdm rich
    )
    model.save(os.path.join(args.model_dir, "g1_self_balance_final"))


if __name__ == "__main__":
    main()
