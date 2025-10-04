"""
Mario Tools - Test models and continue training
"""

import json
import os
import zipfile

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import VecVideoRecorder

from mario_grid import make_mario_env


def get_latest_checkpoint(exp_dir):
    """Find the latest checkpoint in an experiment directory"""
    checkpoint_dir = os.path.join(exp_dir, "models", "checkpoints")
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".zip") and "mario_PPO_" in file:
            steps = int(file.split("_")[2])
            checkpoints.append((steps, os.path.join(checkpoint_dir, file)))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def read_checkpoint_info(checkpoint_path):
    """Read training information from checkpoint"""
    with zipfile.ZipFile(checkpoint_path, "r") as zf:
        data = json.loads(zf.read("data").decode("utf-8"))
        info = {
            "current_timesteps": data.get("num_timesteps", 0),
            "total_timesteps": data.get("_total_timesteps", 40000000),
            "n_envs": data.get("n_envs", 1),
        }
        info["remaining_timesteps"] = info["total_timesteps"] - info["current_timesteps"]
        return info


def test_model(model_path, n_eval_episodes=10, video_dir="test_videos", seed=2):
    """Test a trained model with video recording"""
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    os.makedirs(video_dir, exist_ok=True)

    test_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=1,
        seed=seed,
        max_episode_steps=4096,
        normalize_reward=False,
        env_kwargs={"render_mode": "rgb_array"},
    )

    test_env = VecVideoRecorder(
        test_env,
        video_dir,
        record_video_trigger=lambda _: True,
        video_length=10000,
        name_prefix="mario_test",
    )

    episode_rewards, episode_lengths = evaluate_policy(
        model,
        test_env,
        n_eval_episodes=n_eval_episodes,
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
    print(f"Average length: {mean_length:.2f}")
    print(f"Video saved: {video_dir}")


def continue_training(checkpoint_path, additional_timesteps, checkpoint_freq=500000):
    """Continue training from a checkpoint"""
    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path)))
    model_dir = os.path.join(exp_dir, "models")
    log_dir = os.path.join(exp_dir, "logs")

    info = read_checkpoint_info(checkpoint_path)

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Current timesteps: {info['current_timesteps']:,}")
    print(f"Training for: {additional_timesteps:,} more steps")

    train_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=info["n_envs"],
        seed=0,
        max_episode_steps=4096,
        normalize_reward=True,
        monitor_dir=f"{log_dir}/train_continued",
    )

    eval_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=1,
        seed=1,
        max_episode_steps=4096,
        normalize_reward=False,
        monitor_dir=f"{log_dir}/eval_continued",
    )

    model = PPO.load(checkpoint_path, env=train_env, device="cpu")

    current_timesteps = model.num_timesteps
    progress = current_timesteps / 40000000
    remaining_fraction = 1.0 - progress
    current_lr = 4e-4 * (1 - progress) + 2e-4 * progress
    model.learning_rate = LinearSchedule(current_lr, 2e-4, remaining_fraction)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // info["n_envs"],
        save_path=f"{model_dir}/checkpoints",
        name_prefix="mario_PPO",
        verbose=1,
    )

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=20,
        min_evals=10,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_dir}/best_model",
        log_path=f"{log_dir}/eval",
        eval_freq=500000 // info["n_envs"],
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback,
        verbose=1,
    )

    callbacks = [checkpoint_callback, eval_callback]

    print("\nStarting training...")

    model.learn(
        total_timesteps=additional_timesteps,
        callback=callbacks,
        tb_log_name="mario_PPO",
        progress_bar=True,
        reset_num_timesteps=False,
    )

    final_model_path = f"{model_dir}/model_{model.num_timesteps}_steps.zip"
    model.save(final_model_path)
    print(f"\nSaved model: {final_model_path}")
    print(f"Total timesteps: {model.num_timesteps:,}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python mario_tools.py test <model_path> [n_episodes]")
        print("  python mario_tools.py continue <checkpoint_path> <additional_steps>")
    elif sys.argv[1] == "test":
        model_path = sys.argv[2]
        n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        test_model(model_path, n_episodes)
    elif sys.argv[1] == "continue":
        checkpoint_path = sys.argv[2]
        additional_steps = int(sys.argv[3])
        continue_training(checkpoint_path, additional_steps)
