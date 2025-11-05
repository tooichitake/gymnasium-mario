"""
Fine-tune pretrained Mario model using Kickstarting (KS) for knowledge retention.

KS adds an auxiliary KL divergence loss that keeps the fine-tuned policy close to
the pretrained policy, preventing catastrophic forgetting while adapting to the new task.

Reference: "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting
Mitigation Problem" (ICML 2024 Spotlight)
"""

import os

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import explained_variance

from mario import make_mario_env


class PPO_KS(PPO):
    """
    PPO with Kickstarting (KS) auxiliary loss for knowledge retention.

    Adds KL divergence loss: L_KS = E_{s ~ π_θ}[D_KL(π*(s) || π_θ(s))]
    where π* is the pretrained policy and π_θ is the current policy.

    Usage:
        # Load pretrained model
        model = PPO_KS.load(checkpoint_path, env=train_env)

        # Set KS parameters
        model.setup_ks(pretrained_model_path=checkpoint_path, ks_coef=0.01)

        # Override fine-tuning hyperparameters
        model.learning_rate = get_schedule_fn(1e-5)
        model.clip_range = get_schedule_fn(0.1)

        # Train
        model.learn(...)

    :param pretrained_model: Pretrained PPO model for KS (set via setup_ks())
    :param ks_coef: Coefficient for KS auxiliary loss (β in paper)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_model = None
        self.ks_coef = 0.01

    def setup_ks(self, pretrained_model_path: str, ks_coef: float = 0.01):
        """
        Setup KS auxiliary loss by loading pretrained model.

        :param pretrained_model_path: Path to pretrained PPO model (.zip file)
        :param ks_coef: Coefficient for KS auxiliary loss (β in paper)
        """
        self.pretrained_model = PPO.load(pretrained_model_path, device=self.device)
        self.pretrained_model.policy.eval()

        # Freeze pretrained model parameters
        for param in self.pretrained_model.policy.parameters():
            param.requires_grad = False

        self.ks_coef = ks_coef

    def train(self) -> None:
        """
        Update policy using the current rollout buffer, with KS auxiliary loss.

        Overrides PPO.train() to add KL divergence between current policy and
        pretrained policy as a knowledge retention mechanism.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        ks_losses = []  # Track KS losses

        continue_training = True

        # Train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # ==================== Standard PPO Loss ====================
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy gradient loss (clipped)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Value loss
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = th.nn.functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # ==================== KS Auxiliary Loss ====================
                # Compute KL divergence between pretrained and current policy
                with th.no_grad():
                    # Get pretrained policy distribution
                    pretrained_distribution = self.pretrained_model.policy.get_distribution(
                        rollout_data.observations
                    )
                    pretrained_log_prob = pretrained_distribution.log_prob(actions)

                # Get current policy distribution
                current_distribution = self.policy.get_distribution(rollout_data.observations)
                current_log_prob = current_distribution.log_prob(actions)

                # KL divergence: D_KL(pretrained || current)
                # = E[log(pretrained) - log(current)]
                ks_loss = th.mean(pretrained_log_prob - current_log_prob)
                ks_losses.append(ks_loss.item())

                # ==================== Combined Loss ====================
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.ks_coef * ks_loss  # Add KS loss
                )

                # Calculate approximate form of reverse KL divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # Logging
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

        # Log KS-specific metrics
        self.logger.record("train/ks_loss", np.mean(ks_losses))
        self.logger.record("train/ks_coef", self.ks_coef)

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())


if __name__ == "__main__":
    # Configuration
    checkpoint_path = "results/ppo/exp13/models/checkpoints/mario_PPO_10000000_steps.zip"
    exp_dir = "results/ppo/exp_ks_1-1"
    model_dir = os.path.join(exp_dir, "models")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Environment
    train_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=8,
        wrapper_kwargs={"frame_skip": 4, "screen_size": 84, "noop_max": 0},
        vec_normalize_kwargs={"training": True, "norm_obs": False, "norm_reward": True, "gamma": 0.99},
        monitor_dir=f"{log_dir}/train",
    )

    # Load pretrained model (all parameters inherited)
    model = PPO_KS.load(checkpoint_path, env=train_env)

    # Setup tensorboard logging
    model.tensorboard_log = os.path.join(log_dir, "tensorboard")

    # Setup KS auxiliary loss
    ks_coef = 0.1
    model.setup_ks(pretrained_model_path=checkpoint_path, ks_coef=ks_coef)

    total_timesteps = 1e7

    print(f"\nKS Fine-tuning: 1-1 | β={ks_coef} | {total_timesteps:.0e} steps")
    print(f"Experiment: {exp_dir}\n")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=12500,
        save_path=f"{model_dir}/checkpoints",
        name_prefix="mario_PPO_KS",
        verbose=1,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback],
        tb_log_name="mario_PPO_KS",
        progress_bar=True,
        reset_num_timesteps=False,
    )

    # Save
    final_model_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_model_path)
    train_env.save(os.path.join(model_dir, "vec_normalize.pkl"))
    train_env.close()

    print(f"\n✓ Saved to: {final_model_path}")
    print(f"✓ Evaluate: python evaluate_forgetting.py --finetuned {final_model_path}")
