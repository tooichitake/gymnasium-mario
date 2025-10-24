"""
Training script for Super Mario Bros PPO agent
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from mario import ImpalaCNN, create_experiment_folder, make_mario_env

if __name__ == "__main__":
    # Create experiment folder
    exp_num, exp_dir, model_dir, log_dir, _ = create_experiment_folder()

    # Create training environment
    train_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=4,
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "use_single_stage_episodes": True,
            "noop_max": 80,
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

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=f"{model_dir}/checkpoints",
        name_prefix="mario_PPO",
        verbose=1,
    )

    # Create PPO model
    model = PPO(
        "CnnPolicy",
        train_env,
        n_steps=4096,
        batch_size=512,
        n_epochs=10,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=1.8e-3,
        clip_range=0.2,
        vf_coef=0.643,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=256,
                channels=[16, 32, 32],
                normalized_image=False,
            ),
            net_arch=[],
        ),
        verbose=0,
        tensorboard_log=f"{log_dir}/tensorboard",
    )

    model.learn(
        total_timesteps=1e7,
        callback=[checkpoint_callback],
        tb_log_name="mario_PPO",
        progress_bar=True,
    )

    train_env.close()
