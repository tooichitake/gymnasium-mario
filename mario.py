"""
Super Mario Bros PPO Training with CNN-based Observations
"""

import os

import gymnasium as gym
import gymnasium_super_mario_bros
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from gymnasium import spaces
from gymnasium_super_mario_bros.actions import SIMPLE_MOVEMENT
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
)


class ImpalaCNN(BaseFeaturesExtractor):
    """
    Deep residual CNN architecture for visual RL tasks.

    Architecture:
        - 3 ConvSequences with channels [16, 32, 32] (base configuration, scale=1)
        - Each ConvSequence: Conv3x3 → MaxPool2d(3×3, stride=2) → 2 ResidualBlocks
        - ResidualBlock: ReLU → Conv3x3 → ReLU → Conv3x3 → Add skip connection
        - Final: Flatten → ReLU → Linear(3872→256)

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted (output of final linear layer, default 256)
    :param channels: List of output channels for each convolutional sequence (default [16, 32, 32])
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        channels: list = None,
        normalized_image: bool = False,
    ) -> None:
        if channels is None:
            channels = [16, 32, 32]

        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        layers = []
        in_channels = n_input_channels

        for out_channels in channels:
            layers.append(self._make_conv_sequence(in_channels, out_channels))
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            if not normalized_image:
                sample_input = sample_input / 255.0
            cnn_output = self.cnn(sample_input)
            n_flatten = cnn_output.reshape(cnn_output.size(0), -1).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Apply orthogonal initialization with sqrt(2) scaling for ReLU networks.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _make_conv_sequence(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Create a convolutional sequence: Conv -> MaxPool -> ResBlock -> ResBlock

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :return: Sequential module containing the conv sequence
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_residual_block(out_channels),
            self._make_residual_block(out_channels),
        )

    def _make_residual_block(self, channels: int) -> nn.Module:
        """
        Create a residual block: ReLU -> Conv -> ReLU -> Conv -> Add

        :param channels: Number of channels (same for input and output)
        :return: ResidualBlock module
        """
        return ResidualBlock(channels)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if observations.dtype == th.uint8:
            observations = observations.float() / 255.0

        features = self.cnn(observations)
        features = features.reshape(features.size(0), -1)
        features = nn.functional.relu(features)
        return self.linear(features)


class ResidualBlock(nn.Module):
    """
    Residual block for IMPALA CNN.

    Architecture: ReLU -> Conv3x3 -> ReLU -> Conv3x3 -> Add residual connection

    :param channels: Number of input and output channels
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        return x + inputs


class NoopResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sample initial states by taking random number of no-ops on reset.

    This increases the diversity of initial states and prevents the agent from
    overfitting to a fixed starting configuration. Particularly useful for Mario
    levels where enemies move even when Mario doesn't.

    :param env: Environment to wrap
    :param noop_max: Maximum value of no-ops to run (default 30)
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipFrame(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        assert env.observation_space.dtype is not None
        assert env.observation_space.shape is not None
        self._obs_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=env.observation_space.dtype
        )
        self._skip = skip

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        terminated = truncated = False
        flag_get_detected = False

        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)

            if info.get("flag_get", False):
                flag_get_detected = True

            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)

            if done or flag_get_detected:
                break

        max_frame = self._obs_buffer.max(axis=0)

        if flag_get_detected:
            info["flag_get"] = True

        return max_frame, total_reward, terminated, truncated, info


class MarioWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Super Mario Bros preprocessing wrapper.

    Applies the following preprocessing steps:

    * Action space reduction: SIMPLE_MOVEMENT (7 actions) by default
    * No-op reset: Random 1-30 no-ops on reset for initial state diversity
    * Frame skipping with max-pooling: 4 by default (handles NES sprite flickering)
    * Resize to square grayscale image: 84x84 by default
    * Optional: Single-stage episode mode


    :param env: Environment to wrap (Super Mario Bros environment)
    :param frame_skip: Frequency at which the agent experiences the game.
        This correspond to repeating the action ``frame_skip`` times.
    :param screen_size: Resize frame to a square image of this size
    :param action_space: Action space to use (default: SIMPLE_MOVEMENT)
    :param noop_max: Maximum number of no-ops on reset (default 30, set to 0 to disable)
    :param use_single_stage_episodes: If True, each episode terminates after completing one stage.
        When False (default), episodes continue across multiple stages
        (e.g., 1-1 → 1-2 → 1-3...) until death or game over.
    """

    def __init__(
        self,
        env: gym.Env,
        frame_skip: int = 4,
        screen_size: int = 84,
        action_space=SIMPLE_MOVEMENT,
        noop_max: int = 30,
        use_single_stage_episodes: bool = False,
    ) -> None:
        env = JoypadSpace(env, action_space)

        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)

        if frame_skip > 1:
            env = MaxAndSkipFrame(env, skip=frame_skip)

        env = WarpFrame(env, width=screen_size, height=screen_size)

        super().__init__(env)
        self._use_single_stage_episodes = use_single_stage_episodes

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self._use_single_stage_episodes and info.get("flag_get", False):
            truncated = True

        return obs, reward, terminated, truncated, info


def make_mario_env(
    env_id="SuperMarioBros-1-1-v0",
    n_envs=1,
    seed=None,
    max_episode_steps=None,
    frame_stack=4,
    wrapper_kwargs=None,
    vec_normalize_kwargs=None,
    env_kwargs=None,
    **kwargs,
):
    """
    Create a wrapped, monitored VecEnv for Super Mario Bros with Atari-style preprocessing.
    Always uses DummyVecEnv (single-process) for consistent behavior.

    Parameters:
        env_id: The environment ID
            - Single stage: "SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0", etc.
            - Random stages: "SuperMarioBrosRandomStages-v0"
        n_envs: Number of environments to create
        seed: Random seed
        max_episode_steps: Maximum episode length in steps (default None, uses environment default of 9999999)
        frame_stack: Number of frames to stack (default 4)
        wrapper_kwargs: Dict of kwargs for MarioWrapper. Defaults:
            - frame_skip: 4
            - screen_size: 84
            - noop_max: 30
            - use_single_stage_episodes: False
        vec_normalize_kwargs: Dict of kwargs for VecNormalize, or None to disable.
            - Pass None: Disable VecNormalize completely (no normalization)
            - Pass dict (can be empty {}): Apply VecNormalize with values merged into defaults
            Default VecNormalize settings:
            - training: True
            - norm_obs: False
            - norm_reward: True
            - clip_obs: 10.0
            - clip_reward: 10.0
            - gamma: 0.982
        env_kwargs: Dict passed to gym.make(), e.g., {"stages": ['1-1', '1-2'], "render_mode": "rgb_array"}
        **kwargs: Additional arguments passed to make_vec_env

    Returns:
        VecEnv: The wrapped vectorized environment with frame stacking and optionally VecNormalize

    Example:
        env = make_mario_env("SuperMarioBros-1-1-v0", n_envs=8)

        env = make_mario_env(
            "SuperMarioBrosRandomStages-v0",
            n_envs=8,
            env_kwargs={"stages": ['1-1', '1-2', '1-3', '1-4']}
        )

        env = make_mario_env(
            "SuperMarioBrosRandomStages-v0",
            n_envs=8,
            env_kwargs={"stages": ['1-1', '1-2', '1-3', '1-4']},
            wrapper_kwargs={"use_single_stage_episodes": True}
        )

        env = make_mario_env("SuperMarioBrosRandomStages-v0", n_envs=8)

        env = make_mario_env(
            "SuperMarioBros-1-1-v0",
            n_envs=8,
            wrapper_kwargs={"frame_skip": 2, "screen_size": 64, "noop_max": 20}
        )

        env = make_mario_env(
            "SuperMarioBros-1-1-v0",
            n_envs=8,
            wrapper_kwargs={"noop_max": 0}
        )

    Available stages (32 total):
        World 1: 1-1, 1-2, 1-3, 1-4
        World 2: 2-1, 2-2, 2-3, 2-4
        World 3: 3-1, 3-2, 3-3, 3-4
        World 4: 4-1, 4-2, 4-3, 4-4
        World 5: 5-1, 5-2, 5-3, 5-4
        World 6: 6-1, 6-2, 6-3, 6-4
        World 7: 7-1, 7-2, 7-3, 7-4
        World 8: 8-1, 8-2, 8-3, 8-4
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    wrapper_kwargs.setdefault("frame_skip", 4)
    wrapper_kwargs.setdefault("screen_size", 84)
    wrapper_kwargs.setdefault("noop_max", 30)
    wrapper_kwargs.setdefault("use_single_stage_episodes", False)

    if env_kwargs is None:
        env_kwargs = {}

    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps

    def wrapper_fn(env):
        return MarioWrapper(env, **wrapper_kwargs)

    vec_env = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_class=wrapper_fn,
        env_kwargs=env_kwargs,
        vec_env_cls=DummyVecEnv,
        **kwargs,
    )

    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)

    vec_env = VecTransposeImage(vec_env)

    # Apply VecNormalize if dict provided (None = disabled)
    if vec_normalize_kwargs is not None:
        default_vec_normalize_kwargs = {
            "training": True,
            "norm_obs": False,
            "norm_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "gamma": 0.982,
        }
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


def evaluate_policy(
    model,
    test_env,
    video_dir=None,
    video_fps=60,
    n_episodes=1,
    deterministic=True,
    record_video=True,
    save_results=True,
    results_path=None,
    model_dir=None,
):
    """
    Evaluate policy, optionally record video and save results.
    Automatically loads best_model if available.

    :param model: Trained model
    :param test_env: Test environment (created externally with render_mode='rgb_array' if recording video)
    :param video_dir: Directory to save video (will be named mario_gameplay.mp4)
    :param video_fps: Video framerate (default 60)
    :param n_episodes: Number of episodes to evaluate (default 1)
    :param deterministic: Use deterministic policy (default True)
    :param record_video: Whether to record video (default True)
    :param save_results: Whether to save results to CSV (default True)
    :param results_path: Path to save results CSV (required if save_results=True)
    :param model_dir: Directory containing best_model subdirectory (optional)
    :return: Dictionary with statistics
    """
    if model_dir is not None:
        best_model_path = os.path.join(model_dir, "best_model", "best_model.zip")
        if os.path.exists(best_model_path):
            model = PPO.load(best_model_path)

    if record_video:
        if video_dir is None:
            raise ValueError("video_dir must be provided when record_video=True")
        video_path = os.path.join(video_dir, "mario_gameplay.mp4")
    else:
        video_path = None

    captured_frames = []
    step_called_since_reset = [False]

    if record_video:

        def capture_all_frames_in_skip():
            def wrapped_step(self, action):
                step_called_since_reset[0] = True
                total_reward = 0.0
                terminated = truncated = False
                flag_get_detected = False

                for i in range(self._skip):
                    obs, reward, terminated, truncated, info = self.env.step(action)

                    if hasattr(self.env, "render"):
                        frame = self.env.render()
                        if frame is not None:
                            captured_frames.append(frame.copy())

                    if info.get("flag_get", False):
                        flag_get_detected = True

                    done = terminated or truncated
                    if i == self._skip - 2:
                        self._obs_buffer[0] = obs
                    if i == self._skip - 1:
                        self._obs_buffer[1] = obs
                    total_reward += float(reward)

                    if done or flag_get_detected:
                        break

                max_frame = self._obs_buffer.max(axis=0)
                if flag_get_detected:
                    info["flag_get"] = True

                return max_frame, total_reward, terminated, truncated, info

            return wrapped_step

        def capture_noop_reset_frames():
            def wrapped_reset(self, **kwargs):
                is_auto_reset = step_called_since_reset[0]
                should_record_noop = not is_auto_reset

                obs, info = self.env.reset(**kwargs)
                if self.override_num_noops is not None:
                    noops = self.override_num_noops
                else:
                    noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
                assert noops > 0

                internal_reset_occurred = False

                for _ in range(noops):
                    obs, _, terminated, truncated, info = self.env.step(self.noop_action)

                    if should_record_noop and not internal_reset_occurred:
                        if hasattr(self.env, "render"):
                            frame = self.env.render()
                            if frame is not None:
                                captured_frames.append(frame.copy())

                    if terminated or truncated:
                        obs, info = self.env.reset(**kwargs)
                        internal_reset_occurred = True

                step_called_since_reset[0] = False

                return obs, info

            return wrapped_reset

        current_env = test_env.envs[0]
        noop_env = None
        while hasattr(current_env, "env"):
            if current_env.__class__.__name__ == "MaxAndSkipFrame":
                current_env.step = lambda action, env=current_env: capture_all_frames_in_skip()(
                    env, action
                )
            elif current_env.__class__.__name__ == "NoopResetEnv":
                noop_env = current_env
            current_env = current_env.env

        if noop_env is not None:
            noop_env.reset = lambda **kwargs: capture_noop_reset_frames()(noop_env, **kwargs)

    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs = test_env.reset()
        episode_reward = 0
        episode_step = 0

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _ = test_env.step(action)
            episode_reward += reward[0]
            episode_step += 1

            if done[0]:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_step)
                break

    test_env.close()

    if record_video and len(captured_frames) > 0:
        clip = ImageSequenceClip(captured_frames, fps=video_fps)
        clip.write_videofile(video_path)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    if save_results:
        if results_path is None:
            raise ValueError("results_path must be provided when save_results=True")

        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        results_df = pd.DataFrame(
            {
                "Episode": range(1, len(episode_rewards) + 1),
                "Reward": episode_rewards,
                "Length": episode_lengths,
            }
        )

        summary_df = pd.DataFrame(
            {
                "Metric": ["Mean Reward", "Std Reward", "Mean Length"],
                "Value": [f"{mean_reward:.2f}", f"{std_reward:.2f}", f"{mean_length:.2f}"],
            }
        )

        with open(results_path, "w") as f:
            results_df.to_csv(f, index=False)
            f.write("\n")
            summary_df.to_csv(f, index=False)

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
    }


if __name__ == "__main__":
    exp_num, exp_dir, model_dir, log_dir, video_dir = create_experiment_folder()

    train_env = make_mario_env(
        "SuperMarioBrosRandomStages-v0",
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
        env_kwargs={"stages": ['1-1', '1-2', '1-3', '1-4']},
        monitor_dir=f"{log_dir}/train",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"{model_dir}/checkpoints",
        name_prefix="mario_PPO",
        verbose=1,
    )

    callbacks = [checkpoint_callback]

    model = PPO(
        "CnnPolicy",
        train_env,
        n_steps=4096,
        batch_size=64,
        n_epochs=10,
        learning_rate=1.4e-5,
        gamma=0.982,
        gae_lambda=0.901,
        ent_coef=1.81e-3,
        clip_range=0.335,
        vf_coef=0.643,
        max_grad_norm=0.578,
        policy_kwargs=dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=256,
                channels=[16, 32, 32],
                normalized_image=False,
            ),
            net_arch=dict(pi=[256], vf=[256]),
        ),
        verbose=0,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
    )

    model.learn(
        total_timesteps=5e6,
        callback=callbacks,
        tb_log_name="mario_PPO",
        progress_bar=True,
    )

    test_env = make_mario_env(
        "SuperMarioBrosRandomStages-v0",
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "use_single_stage_episodes": False,
            "noop_max": 80,
        },
        vec_normalize_kwargs={
            "training": False,
            "norm_reward": False,
        },
        env_kwargs={"stages": ['1-1', '1-2', '1-3', '1-4'], "render_mode": "rgb_array"},
    )

    evaluate_policy(
        model,
        test_env,
        video_dir=video_dir,
        n_episodes=5,
        results_path=os.path.join(log_dir, "test", "test_results.csv"),
        model_dir=model_dir,
    )
