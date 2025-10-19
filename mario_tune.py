"""
Super Mario Bros PPO Hyperparameter Tuning with Optuna
Train on 1-1 stage with IMPALA CNN architecture
"""

import os

import numpy as np
import optuna
import optunahub
from optuna.pruners import MedianPruner
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from mario import ImpalaCNN, make_mario_env


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
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "use_single_stage_episodes": True,
            "noop_max": 80,
        },
        vec_normalize_kwargs={"training": True, "norm_reward": True, "gamma": gamma},
    )

    eval_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=1,
        seed=1,
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "use_single_stage_episodes": True,
            "noop_max": 80,
        },
        vec_normalize_kwargs=None,
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
        # n_trials=30,
        n_trials=2,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    study_df = study.trials_dataframe()
    results_path = f"{optuna_dir}/study_results.csv"
    study_df.to_csv(results_path, index=False)

    visualize_study(study, optuna_dir)
