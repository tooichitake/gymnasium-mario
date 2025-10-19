"""
Testing script for Super Mario Bros trained models
"""

import os

from stable_baselines3 import PPO

from mario import evaluate_policy, make_mario_env

if __name__ == "__main__":
    # Auto-detect video directory from model path
    path_parts = os.path.normpath(
        "results/ppo/exp1/models/checkpoints/mario_PPO_2000000_steps.zip"
    ).split(os.sep)
    if "results" in path_parts and "ppo" in path_parts:
        results_idx = path_parts.index("results")
        ppo_idx = path_parts.index("ppo")
        if ppo_idx == results_idx + 1 and ppo_idx + 1 < len(path_parts):
            exp_dir = os.path.join(*path_parts[: ppo_idx + 2])
        else:
            exp_dir = None
    else:
        exp_dir = None

    # Load model
    model = PPO.load("results/ppo/exp1/models/checkpoints/mario_PPO_2000000_steps.zip")

    # Create test environment
    test_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "use_single_stage_episodes": True,
            "noop_max": 80,
        },
        vec_normalize_kwargs={
            "training": False,
            "norm_reward": False,
        },
        env_kwargs={"render_mode": "rgb_array"},
    )

    # Create output directories
    if exp_dir is not None:
        os.makedirs(os.path.join(exp_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "logs", "test"), exist_ok=True)
    else:
        os.makedirs("test_videos", exist_ok=True)
        os.makedirs("test_logs", exist_ok=True)

    # Evaluate policy
    evaluate_policy(
        model,
        test_env,
        video_dir=os.path.join(exp_dir, "videos") if exp_dir else "test_videos",
        n_episodes=5,
        deterministic=True,
        results_path=(
            os.path.join(exp_dir, "logs", "test", "test_results.csv")
            if exp_dir
            else os.path.join("test_logs", "test_results.csv")
        ),
    )
