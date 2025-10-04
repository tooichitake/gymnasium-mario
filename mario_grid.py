"""
Super Mario Bros PPO Training with RAM-based Observations
"""

import os
import warnings

import gymnasium as gym
import gymnasium_super_mario_bros
import numpy as np
from gymnasium import spaces
from gymnasium_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    VecVideoRecorder,
)


class MarioWrapper(gym.ObservationWrapper):
    """
    Complete wrapper for Super Mario Bros that combines JoypadSpace and RAM observations.
    """

    def __init__(self, env, crop_dim=[0, 16, 0, 13], n_stack=4, n_skip=4, actions=SIMPLE_MOVEMENT):
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

        self.frames = np.zeros(
            ((self.n_stack - 1) * self.n_skip + 1, self.ny, self.nx), dtype=np.float32
        )
        self.frame_index = 0
        self.buffer_size = (self.n_stack - 1) * self.n_skip + 1

        # Pre-compute enemy map as NumPy array for faster lookups
        self.enemy_map_array = np.full(256, -1, dtype=np.float32)
        # Set continuous ranges using NumPy slicing
        self.enemy_map_array[0x06:0x16] = 1  # 0x06 to 0x15
        self.enemy_map_array[0x24:0x28] = 0  # 0x24 to 0x27
        # Set individual values
        self.enemy_map_array[[0x2D, 0x35, 0x37]] = 1
        self.enemy_map_array[[0x2E, 0x30]] = 0

        # Pre-compute RAM addresses for faster access
        self.enemy_indices = np.arange(5, dtype=np.int32)
        self.enemy_type_addresses = 0xF + self.enemy_indices
        self.enemy_ex_high_addresses = 0x6E + self.enemy_indices
        self.enemy_ex_low_addresses = 0x87 + self.enemy_indices
        self.enemy_ey_addresses = 0xCF + self.enemy_indices
        self.mario_addresses = np.array([0x6D, 0x86, 0x03B8, 0x71C], dtype=np.int32)
        self.camera_address = np.int32(0x71C)

        # Pre-compute offsets for stacking: [0, -4, -8, -12]
        self.stack_offsets = -np.arange(self.n_stack, dtype=np.int32) * self.n_skip

        # Pre-allocate grid buffer
        self.grid_buffer = np.empty((13, 16), dtype=np.float32)

    def _get_ram_grid(self):
        """Convert RAM to grid representation"""
        base_env = self.env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        full_ram = base_env.ram
        # Use pre-allocated buffer and reset values
        self.grid_buffer.fill(-1)
        grid = self.grid_buffer

        # Vectorized enemy processing using pre-computed addresses
        enemies = full_ram[self.enemy_type_addresses]
        valid_enemies = enemies != 0

        if np.any(valid_enemies):
            # Extract valid enemy data using pre-computed indices
            valid_indices = self.enemy_indices[valid_enemies]
            enemy_types = enemies[valid_enemies]

            # Batch extract all enemy RAM data at once
            ex_high = full_ram[self.enemy_ex_high_addresses[valid_enemies]]
            ex_low = full_ram[self.enemy_ex_low_addresses[valid_enemies]]
            camera_x = full_ram[self.camera_address]
            ex = (ex_high.astype(np.int32) * 0x100 + ex_low - camera_x) % 512
            ey = full_ram[self.enemy_ey_addresses[valid_enemies]]

            x_locs = ex // 16
            y_locs = (ey - 32) // 16

            # Apply bounds check and update grid
            valid_pos = (x_locs >= 0) & (x_locs < 16) & (y_locs >= 0) & (y_locs < 13)

            # Vectorized grid update for valid positions
            valid_indices = np.where(valid_pos)[0]
            if len(valid_indices) > 0:
                grid[y_locs[valid_indices], x_locs[valid_indices]] = self.enemy_map_array[
                    enemy_types[valid_indices]
                ]

        # Mario position using pre-computed addresses
        mario_data = full_ram[self.mario_addresses]
        mx = (mario_data[0].astype(np.int32) * 0x100 + mario_data[1] - mario_data[3]) % 512
        my = mario_data[2].astype(np.int32)  # Convert to int32 to avoid overflow
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

        # Calculate which frames to stack (relative to current position)
        # Offsets: [0, -4, -8, -12] means [current, 4-steps-ago, 8-steps-ago, 12-steps-ago]
        indices = (self.frame_index + self.stack_offsets) % self.buffer_size
        stacked = self.frames[indices].transpose(1, 2, 0)

        # Update index for next frame
        self.frame_index = (self.frame_index + 1) % self.buffer_size

        return stacked

    def reset(self, **kwargs):
        """Reset environment and frame buffer"""
        self.frames = np.zeros(
            ((self.n_stack - 1) * self.n_skip + 1, self.ny, self.nx), dtype=np.float32
        )
        self.frame_index = 0

        obs, info = self.env.reset(**kwargs)

        # Initialize all frames with the same grid (more efficient than loop)
        initial_grid = self._get_ram_grid()
        self.frames[:] = initial_grid

        return self.observation(obs), info


def make_mario_env(
    env_id="SuperMarioBros-1-1-v0",
    n_envs=1,
    seed=None,
    max_episode_steps=4096,
    use_vec_normalize=True,
    vec_normalize_kwargs=None,
    **kwargs,
):
    """
    Create a wrapped, monitored VecEnv for Super Mario Bros.
    Uses make_vec_env with Mario-specific wrapper and optional VecNormalize.
    Automatically selects DummyVecEnv for n_envs=1, SubprocVecEnv for n_envs>1.

    Parameters:
        env_id: The environment ID
        n_envs: Number of environments to create
        seed: Random seed
        max_episode_steps: Maximum episode length in steps (default 4096)
        use_vec_normalize: Whether to wrap with VecNormalize (default True)
        vec_normalize_kwargs: Dict of kwargs for VecNormalize. Defaults:
            - training: True
            - norm_obs: False
            - norm_reward: True
            - clip_obs: 10.0
            - clip_reward: 10.0
            - gamma: 0.99
        **kwargs: Additional arguments passed to make_vec_env

    Returns:
        VecEnv: The wrapped vectorized environment (with VecNormalize if enabled)
    """
    wrapper_kwargs = kwargs.pop("wrapper_kwargs", {})

    # Extract env_kwargs and add max_episode_steps
    env_kwargs = kwargs.pop("env_kwargs", {})
    env_kwargs["max_episode_steps"] = max_episode_steps

    def apply_mario_wrapper(env):
        return MarioWrapper(
            env,
            crop_dim=wrapper_kwargs.get("crop_dim", [0, 16, 0, 13]),
            n_stack=wrapper_kwargs.get("n_stack", 4),
            n_skip=wrapper_kwargs.get("n_skip", 4),
            actions=wrapper_kwargs.get("actions", SIMPLE_MOVEMENT),
        )

    # Use DummyVecEnv for single environment, SubprocVecEnv for multiple
    vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv

    vec_env = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_class=apply_mario_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        **kwargs,
    )

    # Add VecNormalize wrapper if requested
    if use_vec_normalize:
        # Set default VecNormalize parameters
        default_vec_normalize_kwargs = {
            "training": True,
            "norm_obs": False,
            "norm_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "gamma": 0.99,
        }
        # Update with user-provided kwargs
        if vec_normalize_kwargs is not None:
            default_vec_normalize_kwargs.update(vec_normalize_kwargs)

        vec_env = VecNormalize(vec_env, **default_vec_normalize_kwargs)

    return vec_env


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
    # Suppress false-positive version warning (1-1 is level name, not version)
    warnings.filterwarnings(
        "ignore", message=".*SuperMarioBros.*out of date.*", category=DeprecationWarning
    )

    exp_num, exp_dir, model_dir, log_dir, video_dir = create_experiment_folder()

    # Create training environment with VecNormalize for reward normalization
    train_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=1,
        seed=0,
        max_episode_steps=4096,
        use_vec_normalize=True,
        vec_normalize_kwargs={"training": True, "norm_reward": True},
        monitor_dir=f"{log_dir}/train",
    )

    # Create evaluation environment with VecNormalize but without updating statistics
    # Normalize rewards using training statistics for consistent evaluation
    eval_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=1,
        seed=1,
        max_episode_steps=4096,
        use_vec_normalize=True,
        vec_normalize_kwargs={"training": False, "norm_reward": True},
        monitor_dir=f"{log_dir}/eval",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=500000,  # Save every 500k steps
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
        eval_freq=500000,  # Evaluate every 500k steps
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
        n_steps=8192,
        batch_size=2048,
        n_epochs=10,
        learning_rate=LinearSchedule(4e-4, 2e-4, 1.0),
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=LinearSchedule(0.22, 0.18, 1.0),
        vf_coef=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256],
            )
        ),
        verbose=0,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        device="cpu",
    )

    model.learn(
        total_timesteps=3e7,
        # total_timesteps=1e5,
        callback=callbacks,
        tb_log_name="mario_PPO",
        progress_bar=True,
    )

    # Load best model if it exists
    best_model_path = f"{model_dir}/best_model/best_model.zip"
    if os.path.exists(best_model_path):
        test_model = PPO.load(best_model_path)
        print("Loaded best model for testing")
    else:
        test_model = model
        print("Best model not found, using final model")

    # Create test environment without VecNormalize (show raw rewards)
    test_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=1,
        seed=2,
        max_episode_steps=4096,
        use_vec_normalize=False,
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
        test_model,
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
