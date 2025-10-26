"""
Continue training from checkpoint for Super Mario Bros PPO agent
"""

import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from mario import make_mario_env

if __name__ == "__main__":
    checkpoint_path = "results/ppo/exp1/models/checkpoints/mario_PPO_5000000_steps.zip"

    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path)))
    model_dir = os.path.join(exp_dir, "models")
    log_dir = os.path.join(exp_dir, "logs")

    train_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=8,
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "use_single_stage_episodes": False,
            "noop_max": 0,
        },
        vec_normalize_kwargs={
            "training": True,
            "norm_obs": False,
            "norm_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 50.0,
            "gamma": 0.99,
        },
        monitor_dir=f"{log_dir}/train",
    )

    model = PPO.load(checkpoint_path, env=train_env)

    # Set tensorboard log path for continued training
    model.tensorboard_log = os.path.join(log_dir, "tensorboard")

    checkpoint_callback = CheckpointCallback(
        save_freq=12500,
        save_path=f"{model_dir}/checkpoints",
        name_prefix="mario_PPO",
        verbose=1,
    )

    model.learn(
        total_timesteps=1e7,
        callback=[checkpoint_callback],
        tb_log_name="mario_PPO",
        progress_bar=True,
        reset_num_timesteps=False,
    )

    train_env.close()
