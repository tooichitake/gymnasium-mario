"""
Continue training from checkpoint for Super Mario Bros PPO agent
"""

import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from mario import make_mario_env

if __name__ == "__main__":
    # Define checkpoint path once
    checkpoint_path = "results/ppo/exp4/models/checkpoints/mario_PPO_2600000_steps.zip"

    # Extract experiment directory from checkpoint path
    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path)))
    model_dir = os.path.join(exp_dir, "models")
    log_dir = os.path.join(exp_dir, "logs")

    # Create training environment (match original training config)
    train_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=4,
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "use_single_stage_episodes": False,
            "noop_max": 80,
        },
        vec_normalize_kwargs={
            "training": True,
            "norm_obs": False,
            "norm_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "gamma": 0.982,
        },
        monitor_dir=f"{log_dir}/train",
    )

    # Load model
    model = PPO.load(checkpoint_path, env=train_env)

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=f"{model_dir}/checkpoints",
        name_prefix="mario_PPO",
        verbose=1,
    )

    # Continue training
    model.learn(
        total_timesteps=1e7,
        callback=[checkpoint_callback],
        tb_log_name="mario_PPO",
        progress_bar=True,
        reset_num_timesteps=False,
    )

    train_env.close()
