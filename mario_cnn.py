"""
Super Mario Bros PPO Training with CNN-based Observations
"""

import os

import gymnasium as gym
import gymnasium_super_mario_bros
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from gymnasium_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
    VecVideoRecorder,
)


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN architecture for image-based observations.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(
            observation_space, check_channels=False, normalized_image=normalized_image
        ), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        # Apply orthogonal initialization with sqrt(2) scaling
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

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class ImpalaCNN(BaseFeaturesExtractor):
    """
    Deep residual CNN architecture for visual RL tasks.

    Architecture:
        - 3 ConvSequences with depths [16, 32, 32]
        - Each ConvSequence: Conv3x3 → MaxPool2d(3×3, stride=2) → 2 ResidualBlocks
        - ResidualBlock: ReLU → Conv3x3 → ReLU → Conv3x3 → Add skip connection
        - Final: Flatten → ReLU → Linear(3872→256)

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted (output of final linear layer, default 256)
    :param depths: List of channel depths for each convolutional sequence (default [16, 32, 32])
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        depths: list = None,
        normalized_image: bool = False,
    ) -> None:
        if depths is None:
            depths = [16, 32, 32]

        assert isinstance(observation_space, spaces.Box), (
            "ImpalaCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)

        assert is_image_space(
            observation_space, check_channels=False, normalized_image=normalized_image
        ), (
            "You should use ImpalaCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )

        n_input_channels = observation_space.shape[0]

        # Build IMPALA CNN with conv sequences
        layers = []
        in_channels = n_input_channels

        for depth in depths:
            layers.append(self._make_conv_sequence(in_channels, depth))
            in_channels = depth

        self.cnn = nn.Sequential(*layers)

        # Compute shape by doing one forward pass
        with th.no_grad():
            # Scale to [0, 1] if not normalized
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            if not normalized_image:
                sample_input = sample_input / 255.0
            cnn_output = self.cnn(sample_input)
            # Flatten: (batch, channels, height, width) -> (batch, channels*height*width)
            n_flatten = cnn_output.reshape(cnn_output.size(0), -1).shape[1]

        # Final dense layers
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        # Apply orthogonal initialization with sqrt(2) scaling
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
        # Normalize images from [0, 255] to [0, 1] if uint8
        if observations.dtype == th.uint8:
            observations = observations.float() / 255.0

        # Pass through CNN
        features = self.cnn(observations)

        # Flatten
        features = features.reshape(features.size(0), -1)

        # ReLU after flatten (matching OpenAI Baselines)
        features = nn.functional.relu(features)

        # Pass through linear layers
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
        # most recent raw observations (for max pooling across time steps)
        assert (
            env.observation_space.dtype is not None
        ), "No dtype specified for the observation space"
        assert env.observation_space.shape is not None, "No shape defined for the observation space"
        self._obs_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=env.observation_space.dtype
        )
        self._skip = skip

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step the environment with the given action.
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info


class MarioWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Super Mario Bros preprocessing wrapper.

    Applies the following preprocessing steps:

    * Action space reduction: SIMPLE_MOVEMENT (7 actions) by default
    * Frame skipping with max-pooling: 4 by default (handles NES sprite flickering)
    * Resize to square grayscale image: 84x84 by default

    This wrapper is inspired by ``stable_baselines3.common.atari_wrappers.AtariWrapper``
    but adapted for Super Mario Bros specific requirements.

    .. note::
        Mario doesn't have lives system like Atari games, so EpisodicLifeEnv is not needed.
        Mario also doesn't require FIRE action on reset or noop resets.

    :param env: Environment to wrap (Super Mario Bros environment)
    :param frame_skip: Frequency at which the agent experiences the game.
        This correspond to repeating the action ``frame_skip`` times.
    :param screen_size: Resize frame to a square image of this size
    :param action_space: Action space to use (default: SIMPLE_MOVEMENT)
    """

    def __init__(
        self,
        env: gym.Env,
        frame_skip: int = 4,
        screen_size: int = 84,
        action_space=SIMPLE_MOVEMENT,
    ) -> None:
        # Apply action space wrapper
        env = JoypadSpace(env, action_space)

        # frame_skip=1 is the same as no frame-skip (action repeat)
        if frame_skip > 1:
            env = MaxAndSkipFrame(env, skip=frame_skip)

        # Apply grayscale and resize wrapper
        env = WarpFrame(env, width=screen_size, height=screen_size)

        super().__init__(env)


def make_mario_env(
    env_id="SuperMarioBros-1-1-v0",
    n_envs=1,
    seed=None,
    max_episode_steps=None,
    frame_stack=4,
    use_vec_normalize=True,
    wrapper_kwargs=None,
    vec_normalize_kwargs=None,
    env_kwargs=None,
    **kwargs,
):
    """
    Create a wrapped, monitored VecEnv for Super Mario Bros with Atari-style preprocessing.
    Automatically selects DummyVecEnv for n_envs=1, SubprocVecEnv for n_envs>1.

    Parameters:
        env_id: The environment ID
            - Single stage: "SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0", etc.
            - Random stages: "SuperMarioBrosRandomStages-v0"
        n_envs: Number of environments to create
        seed: Random seed
        max_episode_steps: Maximum episode length in steps (default None, uses environment default of 9999999)
        frame_stack: Number of frames to stack (default 4)
        use_vec_normalize: Whether to wrap with VecNormalize (default True)
        wrapper_kwargs: Dict of kwargs for MarioWrapper. Defaults:
            - frame_skip: 4
            - screen_size: 84
        vec_normalize_kwargs: Dict of kwargs for VecNormalize. Defaults:
            - training: True
            - norm_obs: False
            - norm_reward: True
            - clip_obs: 10.0
            - clip_reward: 10.0
            - gamma: 0.99
        env_kwargs: Dict passed to gym.make(), e.g., {"stages": ['1-1', '1-2'], "render_mode": "rgb_array"}
        **kwargs: Additional arguments passed to make_vec_env

    Returns:
        VecEnv: The wrapped vectorized environment with frame stacking and optional VecNormalize

    Example:
        # Single stage training
        env = make_mario_env("SuperMarioBros-1-1-v0", n_envs=8)

        # Random stage training - World 1 only
        env = make_mario_env(
            "SuperMarioBrosRandomStages-v0",
            n_envs=8,
            env_kwargs={"stages": ['1-1', '1-2', '1-3', '1-4']}
        )

        # Random stage training - All 32 stages (no env_kwargs needed)
        env = make_mario_env("SuperMarioBrosRandomStages-v0", n_envs=8)

        # Custom frame skip and screen size
        env = make_mario_env(
            "SuperMarioBros-1-1-v0",
            n_envs=8,
            wrapper_kwargs={"frame_skip": 2, "screen_size": 64}
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
    # Set default wrapper_kwargs
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    wrapper_kwargs.setdefault("frame_skip", 4)
    wrapper_kwargs.setdefault("screen_size", 84)

    # Set default env_kwargs
    if env_kwargs is None:
        env_kwargs = {}

    # Only set max_episode_steps if explicitly provided
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps

    # Create a wrapper function with the specified wrapper_kwargs
    def wrapper_fn(env):
        return MarioWrapper(env, **wrapper_kwargs)

    # Use DummyVecEnv for single environment, SubprocVecEnv for multiple
    vec_env_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv

    # Create base vectorized environment with custom wrapper
    vec_env = make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_class=wrapper_fn,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        **kwargs,
    )

    # Apply frame stacking
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)

    # Transpose image channels for CNN policy
    vec_env = VecTransposeImage(vec_env)

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
    exp_num, exp_dir, model_dir, log_dir, video_dir = create_experiment_folder()

    # Create training environment with VecNormalize for reward normalization
    # Train on all 32 stages for maximum generalization
    train_env = make_mario_env(
        "SuperMarioBrosRandomStages-v0",
        n_envs=8,
        seed=0,
        frame_stack=4,
        wrapper_kwargs={"frame_skip": 4, "screen_size": 84},
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
        wrapper_kwargs={"frame_skip": 4, "screen_size": 84},
        use_vec_normalize=True,
        vec_normalize_kwargs={"training": False, "norm_reward": True},
        monitor_dir=f"{log_dir}/eval",
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
        n_eval_episodes=1,  # Deterministic env → only need 1 eval
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback,
        verbose=1,
    )

    callbacks = [checkpoint_callback, eval_callback]

    model = PPO(
        "CnnPolicy",
        train_env,
        n_steps=4096,
        batch_size=8192,
        n_epochs=10,
        learning_rate=LinearSchedule(3e-4, 0, 1.0),
        gae_lambda=0.95,
        ent_coef=0.005,
        clip_range=LinearSchedule(0.20, 0, 1.0),
        vf_coef=0.5,
        # Option 1: Nature CNN (Mnih et al. 2015)
        # policy_kwargs=dict(
        #     features_extractor_class=NatureCNN,
        #     features_extractor_kwargs=dict(features_dim=512, normalized_image=False),
        #     net_arch=dict(pi=[512, 512], vf=[512, 512]),
        # ),
        # Option 2: IMPALA CNN (Espeholt et al. 2018)
        policy_kwargs=dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=256, depths=[16, 32, 32], normalized_image=False
            ),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        verbose=0,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
    )

    model.learn(
        total_timesteps=5e7,
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
        wrapper_kwargs={"frame_skip": 4, "screen_size": 84},
        use_vec_normalize=False,
        env_kwargs={"render_mode": "rgb_array"},
    )

    test_env = VecVideoRecorder(
        test_env,
        video_dir,
        record_video_trigger=lambda _: True,
        video_length=10000,
        name_prefix="mario_PPO_gameplay",
    )

    episode_rewards, episode_lengths = evaluate_policy(
        test_model,
        test_env,
        n_eval_episodes=1,
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
