# mario_a2c_naturecnn.py
"""
Super Mario Bros A2C with NatureCNN + Atari-style preprocessing.
- Train on all stages (RandomStages-v0) with 8 parallel envs
- Joypad (SIMPLE_MOVEMENT) + frame-skip(4) + grayscale 84x84 + frame-stack(4)
- VecNormalize (reward norm) for train; eval uses frozen stats
- NatureCNN (features_dim=512) + pi/vf MLP heads [512, 512]
- Checkpoints + periodic eval every 62.5k env steps (like the example)
- Video recording using rgb_array
"""

import os
import time
from typing import Callable, Optional

import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
import gymnasium_super_mario_bros  # registers SuperMarioBros-* ids
from gymnasium import spaces
from gymnasium_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from stable_baselines3 import A2C
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    VecVideoRecorder,
    VecMonitor,
    VecNormalize,
)


# -----------------------------
# NatureCNN feature extractor
# -----------------------------
class NatureCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False):
        assert isinstance(observation_space, spaces.Box), "NatureCNN requires Box observation space"
        super().__init__(observation_space, features_dim)

        assert is_image_space(
            observation_space, check_channels=False, normalized_image=normalized_image
        ), "NatureCNN requires image observations"

        n_ch = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(obs))


# -----------------------------
# Preprocessing wrappers
# -----------------------------
class MaxAndSkipFrame(gym.Wrapper):
    """Repeat action and take max of last two frames (NES flicker)."""
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        assert env.observation_space.dtype is not None and env.observation_space.shape is not None
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class MarioWrapper(gym.Wrapper):
    """SIMPLE_MOVEMENT → frameskip(4) → WarpFrame(84x84 gray)."""
    def __init__(self, env: gym.Env, frame_skip: int = 4, screen_size: int = 84):
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        if frame_skip > 1:
            env = MaxAndSkipFrame(env, skip=frame_skip)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        super().__init__(env)


# -----------------------------
# VecEnv factory (with VecNormalize like the example)
# -----------------------------
def make_mario_env(
    env_id: str,
    n_envs: int,
    seed: int,
    frame_stack: int = 4,
    frame_skip: int = 4,
    screen_size: int = 84,
    use_vec_normalize: bool = True,
    vec_normalize_kwargs: Optional[dict] = None,  # {"training": True/False, "norm_reward": True}
    monitor_dir: Optional[str] = None,
    env_kwargs: Optional[dict] = None,  # {"render_mode": "rgb_array"}
):
    """
    Builds a vectorized env that matches the example’s preprocessing and VecNormalize behavior.
    """
    if env_kwargs is None:
        env_kwargs = {}
    if vec_normalize_kwargs is None:
        vec_normalize_kwargs = {}

    def make_one(rank: int):
        def _thunk():
            env = gym.make(env_id, **env_kwargs)
            env = MarioWrapper(env, frame_skip=frame_skip, screen_size=screen_size)
            if monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
                from stable_baselines3.common.monitor import Monitor
                env = Monitor(env, filename=os.path.join(monitor_dir, f"env_{rank}.monitor.csv"))
            env.reset(seed=seed + rank)
            return env
        return _thunk

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = vec_cls([make_one(i) for i in range(n_envs)])
    venv = VecMonitor(venv)
    venv = VecFrameStack(venv, n_stack=frame_stack)
    venv = VecTransposeImage(venv)  # -> (C,H,W) for CNN

    if use_vec_normalize:
        defaults = dict(
            training=True,
            norm_obs=False,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )
        defaults.update(vec_normalize_kwargs)
        venv = VecNormalize(venv, **defaults)

    return venv


# -----------------------------
# Main
# -----------------------------
def main():
    # === Config (match example values) ===
    ENV_TRAIN = "SuperMarioBrosRandomStages-v0"
    ENV_EVAL  = "SuperMarioBros-1-1-v0"

    N_ENVS_TRAIN = 8
    N_ENVS_EVAL = 1
    SEED_TRAIN = 0
    SEED_EVAL = 1

    TOTAL_TIMESTEPS = int(2e7)     # follow example scale
    N_STEPS = 4096                 # example used 4096
    LEARNING_RATE = 3e-4           # example value
    GAE_LAMBDA = 0.95
    ENT_COEF = 0.005               # example value
    VF_COEF = 0.5
    GAMMA = 0.99
    MAX_GRAD_NORM = 0.5
    DEVICE = "cpu" if th.cuda.is_available() else "cpu"

    LOG_ROOT = os.path.join("results", "a2c_sb3_naturecnn")
    MODEL_DIR = os.path.join(LOG_ROOT, "models")
    BEST_DIR  = os.path.join(MODEL_DIR, "best_model")
    CKPT_DIR  = os.path.join(MODEL_DIR, "checkpoints")
    VIDEO_DIR = os.path.join(LOG_ROOT, "videos")
    for d in [LOG_ROOT, MODEL_DIR, BEST_DIR, CKPT_DIR, VIDEO_DIR]:
        os.makedirs(d, exist_ok=True)

    # === Environments ===
    print(f"Creating {N_ENVS_TRAIN} training envs…")
    train_env = make_mario_env(
        ENV_TRAIN,
        n_envs=N_ENVS_TRAIN,
        seed=SEED_TRAIN,
        frame_stack=4,
        frame_skip=4,
        screen_size=84,
        use_vec_normalize=True,
        vec_normalize_kwargs={"training": True, "norm_reward": True},
        monitor_dir=os.path.join(LOG_ROOT, "logs", "train"),
    )

    print("Creating eval env…")
    eval_env = make_mario_env(
        ENV_EVAL,
        n_envs=N_ENVS_EVAL,
        seed=SEED_EVAL,
        frame_stack=4,
        frame_skip=4,
        screen_size=84,
        use_vec_normalize=True,  # use stats but freeze updates
        vec_normalize_kwargs={"training": False, "norm_reward": True},
        monitor_dir=os.path.join(LOG_ROOT, "logs", "eval"),
    )

    # === Model: A2C + NatureCNN (like example NatureCNN option) ===
    policy_kwargs = dict(
        features_extractor_class=NatureCNN,
        features_extractor_kwargs=dict(features_dim=512, normalized_image=False),
        net_arch=dict(pi=[512, 512], vf=[512, 512]),
    )

    model = A2C(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        tensorboard_log=os.path.join(LOG_ROOT, "tensorboard"),
        policy_kwargs=policy_kwargs,
        device=DEVICE,
        seed=SEED_TRAIN,
        verbose=1,
    )

    # === Callbacks (every 62.5k env steps, like example 500k/8) ===
    save_eval_every = 100000

    checkpoint_cb = CheckpointCallback(
        save_freq=save_eval_every,
        save_path=CKPT_DIR,
        name_prefix="mario_A2C_NatureCNN",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=20,  # like example spirit
        min_evals=10,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path=os.path.join(LOG_ROOT, "eval"),
        eval_freq=save_eval_every,
        n_eval_episodes=1,   # deterministic eval like example
        deterministic=True,
        render=False,
        callback_after_eval=stop_cb,
        verbose=1,
    )

    # === Train ===
    print(f"Using device: {model.device}")
    print(f"Starting A2C (NatureCNN) | env={ENV_TRAIN} | total_timesteps={TOTAL_TIMESTEPS:,} | n_envs={N_ENVS_TRAIN}")
    t0 = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )
    print(f"Training time: {time.time() - t0:.1f}s")

    # Save final
    final_path = os.path.join(MODEL_DIR, "a2c_naturecnn_final")
    model.save(final_path)
    print(f"💾 Saved final model to: {final_path}.zip")

    # === Load best (if any) & record video ===
    to_play = model
    best_zip = os.path.join(BEST_DIR, "best_model.zip")
    if os.path.exists(best_zip):
        to_play = A2C.load(best_zip, env=None, device=DEVICE)
        print("Loaded best A2C model for video playback")

    # Build a video env that renders frames
    video_env = make_mario_env(
        env_id="SuperMarioBros-1-1-v0",
        n_envs=1,
        seed=SEED_EVAL + 100,
        frame_stack=4,
        frame_skip=1,
        screen_size=84,
        use_vec_normalize=False,                  # raw rewards for video
        env_kwargs={"render_mode": "rgb_array"},  # critical for VecVideoRecorder
    )

    video_env = VecVideoRecorder(
        video_env,
        video_folder=VIDEO_DIR,
        record_video_trigger=lambda step: step == 0,  # record first episode
        video_length=10_000,
        name_prefix="mario_A2C_NatureCNN",
    )

    obs = video_env.reset()
    done = [False]
    while not done[0]:
        action, _ = to_play.predict(obs, deterministic=True)
        obs, reward, done, info = video_env.step(action)
    video_env.close()
    print(f"🎥 Saved gameplay video(s) to: {VIDEO_DIR}")

    # Quick evaluation (frozen normalize stats)
    eval_env_quick = make_mario_env(
        ENV_EVAL,
        n_envs=1,
        seed=SEED_EVAL + 123,
        frame_stack=4,
        frame_skip=4,
        screen_size=84,
        use_vec_normalize=True,
        vec_normalize_kwargs={"training": False, "norm_reward": True},
    )
    mean_r, std_r = evaluate_policy(to_play, eval_env_quick, n_eval_episodes=5, deterministic=True)
    print(f"[Eval x5] mean_reward={mean_r:.2f} ± {std_r:.2f}")


if __name__ == "__main__":
    main()
