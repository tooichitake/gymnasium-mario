"""
Super Mario Bros PPO Training with RAM-based Observations
"""

import os

import gymnasium as gym
import gymnasium_super_mario_bros
import numpy as np
from gymnasium import spaces
from gymnasium_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement)
# from sbx import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import VecVideoRecorder


class MarioWrapper(gym.ObservationWrapper):
    """
    Complete wrapper for Super Mario Bros that combines JoypadSpace and RAM observations.
    """

    def __init__(
        self, env, crop_dim=[0, 16, 0, 13], n_stack=2, n_skip=4, actions=SIMPLE_MOVEMENT
    ):
        env = JoypadSpace(env, actions)
        super().__init__(env)

        self.crop_dim = crop_dim
        self.n_stack = n_stack
        self.n_skip = n_skip
        self.nx = crop_dim[1] - crop_dim[0]
        self.ny = crop_dim[3] - crop_dim[2]
        self.observation_space = spaces.Box(
            low=-1, high=2, shape=(self.ny, self.nx, self.n_stack), dtype=np.float32
        )

        self.frames = np.zeros((self.n_skip + 1, self.ny, self.nx), dtype=np.float32)
        self.frame_index = 0

    def _get_ram_grid(self):
        """Convert RAM to grid representation"""
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        full_ram = base_env.ram

        enemy_map = {
            0x00: -1,
            0x01: -1,
            0x02: -1,
            0x03: -1,
            0x04: -1,
            0x05: -1,
            0x06: 1,
            0x07: 1,
            0x08: 1,
            0x09: 1,
            0x0A: 1,
            0x0B: 1,
            0x0C: 1,
            0x0D: 1,
            0x0E: 1,
            0x0F: 1,
            0x10: 1,
            0x11: 1,
            0x12: 1,
            0x13: 1,
            0x14: 1,
            0x15: 1,
            0x16: -1,
            0x17: -1,
            0x18: -1,
            0x19: -1,
            0x1A: -1,
            0x1B: -1,
            0x1C: -1,
            0x1D: -1,
            0x1E: -1,
            0x1F: -1,
            0x20: -1,
            0x21: -1,
            0x22: -1,
            0x23: -1,
            0x24: 0,
            0x25: 0,
            0x26: 0,
            0x27: 0,
            0x28: -1,
            0x29: -1,
            0x2A: -1,
            0x2B: -1,
            0x2C: -1,
            0x2D: 1,
            0x2E: 0,
            0x2F: -1,
            0x30: 0,
            0x31: -1,
            0x32: -1,
            0x33: -1,
            0x34: -1,
            0x35: 1,
            0x36: -1,
            0x37: 1,
            0x38: -1,
            0x39: -1,
            0x3A: -1,
            0x3B: -1,
            0x3C: -1,
            0x3D: -1,
            0x3E: -1,
        }

        grid = -np.ones((13, 16), dtype=np.float32)

        for i in range(5):
            enemy = int(full_ram[0xF + i])
            if enemy != 0:
                ex = int(full_ram[0x6E + i]) * 0x100 + int(full_ram[0x87 + i])
                ey = int(full_ram[0xCF + i])
                ex = (ex - int(full_ram[0x71C])) % 512

                x_loc = ex // 16
                y_loc = (ey - 32) // 16
                if 0 <= x_loc < 16 and 0 <= y_loc < 13:
                    grid[y_loc, x_loc] = enemy_map.get(enemy, -1)

        mx = int(full_ram[0x6D]) * 0x100 + int(full_ram[0x86])
        my = int(full_ram[0x03B8])
        mx = (mx - int(full_ram[0x71C])) % 512
        x_loc = mx // 16
        y_loc = (my - 32) // 16
        if 0 <= x_loc < 16 and 0 <= y_loc < 13:
            grid[y_loc, x_loc] = 2

        return self._crop_grid(grid)

    def _crop_grid(self, grid):
        """Crop grid to specified dimensions"""
        [x0, x1, y0, y1] = self.crop_dim
        return grid[y0:y1, x0:x1]

    def observation(self, obs):
        """Convert observation to RAM grid"""
        self.frames[self.frame_index] = self._get_ram_grid()
        self.frame_index = (self.frame_index + 1) % (self.n_skip + 1)
        stacked = np.zeros((self.ny, self.nx, self.n_stack), dtype=np.float32)
        for i in range(self.n_stack):
            stacked[:, :, i] = self.frames[i * self.n_skip]

        return stacked

    def reset(self, **kwargs):
        """Reset environment and frame buffer"""
        self.frames = np.zeros((self.n_skip + 1, self.ny, self.nx), dtype=np.float32)
        self.frame_index = 0

        obs, info = self.env.reset(**kwargs)

        for i in range(self.n_skip + 1):
            self.frames[i] = self._get_ram_grid()

        return self.observation(obs), info


def make_mario_env(env_id="SuperMarioBros-1-1-v0", n_envs=1, seed=None, **kwargs):
    """
    Create a wrapped, monitored VecEnv for Super Mario Bros.
    Uses make_vec_env with Mario-specific wrapper.

    Parameters:
        env_id: The environment ID
        n_envs: Number of environments to create
        seed: Random seed
        **kwargs: Additional arguments passed to make_vec_env

    Returns:
        VecEnv: The wrapped vectorized environment
    """
    wrapper_kwargs = kwargs.pop("wrapper_kwargs", {})

    def apply_mario_wrapper(env):
        return MarioWrapper(
            env,
            crop_dim=wrapper_kwargs.get("crop_dim", [0, 16, 0, 13]),
            n_stack=wrapper_kwargs.get("n_stack", 2),
            n_skip=wrapper_kwargs.get("n_skip", 4),
            actions=wrapper_kwargs.get("actions", SIMPLE_MOVEMENT),
        )

    return make_vec_env(
        env_id, n_envs=n_envs, seed=seed, wrapper_class=apply_mario_wrapper, **kwargs
    )


def create_experiment_folder(base_dir="results/ppo"):
    """Create experiment folder structure."""
    os.makedirs(base_dir, exist_ok=True)

    exp_num = 1
    while os.path.exists(os.path.join(base_dir, f"exp{exp_num}")):
        exp_num += 1

    exp_dir = os.path.join(base_dir, f"exp{exp_num}")
    model_dir = os.path.join(exp_dir, "models")
    log_dir = os.path.join(exp_dir, "logs")
    video_dir = os.path.join(exp_dir, "videos")

    os.makedirs(model_dir)
    os.makedirs(log_dir)
    os.makedirs(video_dir)

    print(f"Created Experiment {exp_num}: {exp_dir}")

    return exp_num, exp_dir, model_dir, log_dir, video_dir


if __name__ == "__main__":
    exp_num, exp_dir, model_dir, log_dir, video_dir = create_experiment_folder()

    train_env = make_mario_env(
        "SuperMarioBros-1-1-v0", n_envs=8, seed=0, monitor_dir=f"{log_dir}/train"
    )

    eval_env = make_mario_env(
        "SuperMarioBros-1-1-v0", n_envs=1, seed=1, monitor_dir=f"{log_dir}/eval"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=62500,  # Save every 500k steps (500k / 8 envs = 62.5k)
        save_path=f"{model_dir}/checkpoints",
        name_prefix="mario_PPO",
        verbose=1,
    )

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=20,  # Stop after 10M steps without improvement (20 * 500k = 10M)
        min_evals=10,  # Don't stop in first 5M steps (10 * 500k = 5M)
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model",
        log_path=f"{log_dir}/eval",
        eval_freq=62500,  # Evaluate every 500k steps (500k / 8 envs = 62.5k)
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback,
        verbose=1,
    )

    callbacks = [checkpoint_callback, eval_callback]
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=LinearSchedule(2.5e-4, 0, 1.0),
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.001,
        clip_range=LinearSchedule(0.2, 0, 1.0),
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        device="cpu",
    )

    model.learn(
        total_timesteps=1e7,
        callback=callbacks,
        tb_log_name="mario_PPO",
        progress_bar=True,
    )

    test_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=1,
        seed=2,
        env_kwargs={"render_mode": "rgb_array"},
    )

    test_env = VecVideoRecorder(
        test_env,
        video_dir,
        record_video_trigger=lambda x: True,
        video_length=10000,
        name_prefix="mario_PPO_gameplay",
    )

    episode_rewards, episode_lengths = evaluate_policy(
        model,
        test_env,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        return_episode_rewards=True,
    )

    test_env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    print(f"\nPerformance Summary:")
    print(f"Average reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Average episode length: {mean_length:.2f}")
    print(f"\nDetailed results:")
    for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths)):
        print(f"Episode {i + 1}: Reward = {reward}, Length = {length}")
    print(f"\nVideo saved in: {video_dir}")
