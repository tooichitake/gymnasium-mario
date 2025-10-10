"""
Super Mario Bros PPO Hyperparameter Tuning with Optuna
Train on 1-1 stage with IMPALA CNN architecture
"""

import os

import gymnasium as gym
import gymnasium_super_mario_bros
import numpy as np
import optuna
import optunahub
import torch as th
import torch.nn as nn
from gymnasium import spaces
from gymnasium_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from optuna.pruners import MedianPruner
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
)


class ImpalaCNN(BaseFeaturesExtractor):
    """
    Deep residual CNN architecture for visual RL tasks (IMPALA, Espeholt et al. 2018).

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

        for out_channels in channels:
            layers.append(self._make_conv_sequence(in_channels, out_channels))
            in_channels = out_channels

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


class OptunaPruningCallback(BaseCallback):
    """
    Callback for Optuna pruning during training.
    Reports intermediate values to Optuna, saves best model, and raises TrialPruned if trial should be pruned.
    """

    def __init__(self, trial, eval_env, eval_freq=50000, n_eval_episodes=1):
        super().__init__()
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_count = 0
        self.best_reward = -np.inf  # Best reward observed during training

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current model
            eval_reward, _ = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=True
            )

            # Track best reward (no file I/O for hyperparameter tuning)
            if eval_reward > self.best_reward:
                self.best_reward = eval_reward
                print(
                    f"Trial {self.trial.number}: New best reward {eval_reward:.2f} at step {self.n_calls}"
                )

            # Report to Optuna (current evaluation, not best)
            self.eval_count += 1
            self.trial.report(eval_reward, self.eval_count)

            # Check if trial should be pruned
            if self.trial.should_prune():
                print(
                    f"Trial {self.trial.number} pruned at step {self.n_calls} (eval {self.eval_count})"
                )
                raise optuna.TrialPruned()

        return True


def visualize_study(study, output_dir):
    """
    Generate and save all Optuna visualization plots.

    Parameters:
        study: Optuna study object
        output_dir: Directory to save visualization plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Parameter Importance
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(os.path.join(output_dir, "param_importances.html"))

    # 2. Optimization History
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(os.path.join(output_dir, "optimization_history.html"))

    # 3. Parallel Coordinate Plot
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_html(os.path.join(output_dir, "parallel_coordinate.html"))

    # 4. Slice Plot
    fig = optuna.visualization.plot_slice(study)
    fig.write_html(os.path.join(output_dir, "slice.html"))

    # 5. Contour Plot
    fig = optuna.visualization.plot_contour(study)
    fig.write_html(os.path.join(output_dir, "contour.html"))

    # 6. EDF (Empirical Distribution Function) Plot
    fig = optuna.visualization.plot_edf(study)
    fig.write_html(os.path.join(output_dir, "edf.html"))

    # 7. Timeline Plot
    fig = optuna.visualization.plot_timeline(study)
    fig.write_html(os.path.join(output_dir, "timeline.html"))


def objective(trial):
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024, 2048])
    n_epochs = trial.suggest_int("n_epochs", 4, 16)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.05, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.5)
    vf_coef = trial.suggest_float("vf_coef", 0.3, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 0.8)

    # Print trial parameters
    print("\n" + "=" * 80)
    print(f"Trial {trial.number} - Sampled Hyperparameters:")
    print("=" * 80)
    print(f"  n_steps:        {n_steps}")
    print(f"  batch_size:     {batch_size}")
    print(f"  n_epochs:       {n_epochs}")
    print(f"  learning_rate:  {learning_rate:.6e}")
    print(f"  gamma:          {gamma:.4f}")
    print(f"  gae_lambda:     {gae_lambda:.4f}")
    print(f"  ent_coef:       {ent_coef:.6e}")
    print(f"  clip_range:     {clip_range:.4f}")
    print(f"  vf_coef:        {vf_coef:.4f}")
    print(f"  max_grad_norm:  {max_grad_norm:.4f}")
    print("=" * 80 + "\n")

    # Create environments - Train on stage 1-1
    train_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=4,
        seed=0,
        frame_stack=4,
        wrapper_kwargs={"frame_skip": 4, "screen_size": 84},
        use_vec_normalize=True,
        vec_normalize_kwargs={"training": True, "norm_reward": True, "gamma": gamma},
    )

    eval_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=1,
        seed=1,
        wrapper_kwargs={"frame_skip": 4, "screen_size": 84},
        use_vec_normalize=False,
    )

    # Callbacks
    pruning_callback = OptunaPruningCallback(
        trial=trial,
        eval_env=eval_env,
        eval_freq=1e5 // 4,  # 100k steps / n_envs = 25k calls
        n_eval_episodes=1,
    )

    model = PPO(
        "CnnPolicy",
        train_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=256,
                channels=[32, 64, 64],
                normalized_image=False,
            ),
            net_arch=dict(pi=[256], vf=[256]),
        ),
        verbose=0,
    )

    try:
        model.learn(
            total_timesteps=1e6,
            callback=pruning_callback,
            progress_bar=True,
        )
    except optuna.TrialPruned:
        train_env.close()
        eval_env.close()
        raise
    except Exception:
        train_env.close()
        eval_env.close()
        raise

    # Return best reward observed during training (no file I/O needed)
    best_reward = pruning_callback.best_reward

    train_env.close()
    eval_env.close()

    return best_reward


if __name__ == "__main__":
    optuna_dir = "results/optuna/ppo"
    os.makedirs(optuna_dir, exist_ok=True)

    # SQLite storage with timeout to handle concurrent access
    storage_name = f"sqlite:///{optuna_dir}/study.db?timeout=300"

    # Load AutoSampler from OptunaHub
    module = optunahub.load_module("samplers/auto_sampler")
    sampler = module.AutoSampler()

    study = optuna.create_study(
        study_name="mario_ppo_hyperparameter_tuning_impala_cnn",
        direction="maximize",
        sampler=sampler,
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=30,
        show_progress_bar=True,
        gc_after_trial=True,  # Clean up memory after each trial
    )

    study_df = study.trials_dataframe()
    results_path = f"{optuna_dir}/study_results.csv"
    study_df.to_csv(results_path, index=False)

    visualize_study(study, optuna_dir)
