"""
Super Mario Bros PPO Hyperparameter Tuning with Optuna
Train on 1-1 stage
"""

import os

import optuna
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from mario import ImpalaCNN, make_mario_env


def visualize_study(study, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(os.path.join(output_dir, "param_importances.html"))

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(os.path.join(output_dir, "optimization_history.html"))

    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_html(os.path.join(output_dir, "parallel_coordinate.html"))

    fig = optuna.visualization.plot_slice(study)
    fig.write_html(os.path.join(output_dir, "slice.html"))

    fig = optuna.visualization.plot_contour(study)
    fig.write_html(os.path.join(output_dir, "contour.html"))

    fig = optuna.visualization.plot_edf(study)
    fig.write_html(os.path.join(output_dir, "edf.html"))

    fig = optuna.visualization.plot_timeline(study)
    fig.write_html(os.path.join(output_dir, "timeline.html"))


def objective(trial):
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024, 2048])
    n_epochs = trial.suggest_int("n_epochs", 4, 16)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.05, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.5)
    vf_coef = trial.suggest_float("vf_coef", 0.3, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 0.8)

    features_dim = trial.suggest_categorical("features_dim", [256, 512])
    net_arch_type = trial.suggest_categorical("net_arch_type", ["direct", "1layer", "2layer"])

    channels = [32, 64, 64]

    if net_arch_type == "direct":
        net_arch = []
    elif net_arch_type == "1layer":
        net_arch = dict(pi=[256], vf=[256])
    else:
        net_arch = dict(pi=[256, 256], vf=[256, 256])

    print("\n" + "=" * 80)
    print(f"Trial {trial.number}")
    print("=" * 80)
    print(
        f"n_steps={n_steps}, batch_size={batch_size}, n_epochs={n_epochs}, lr={learning_rate:.2e}"
    )
    print(f"gamma={gamma:.3f}, gae_lambda={gae_lambda:.3f}, ent_coef={ent_coef:.2e}")
    print(f"clip_range={clip_range:.2f}, vf_coef={vf_coef:.2f}, max_grad_norm={max_grad_norm:.2f}")
    print(f"features_dim={features_dim}, net_arch_type={net_arch_type}")
    print("=" * 80 + "\n")

    train_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=4,
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
            "gamma": gamma,
        },
    )

    test_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=1,
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "use_single_stage_episodes": False,
            "noop_max": 0,
        },
        vec_normalize_kwargs={
            "training": False,
            "norm_reward": False,
        },
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
                features_dim=features_dim,
                channels=channels,
                normalized_image=False,
            ),
            net_arch=net_arch,
        ),
        verbose=0,
    )

    model.learn(total_timesteps=5e5, progress_bar=True)

    reward, _ = evaluate_policy(model, test_env, n_eval_episodes=1, deterministic=True)

    print(f"\nTrial {trial.number}: {reward:.2f}")

    train_env.close()
    test_env.close()

    return reward


if __name__ == "__main__":
    optuna_dir = "results/optuna/ppo"
    os.makedirs(optuna_dir, exist_ok=True)

    storage_name = f"sqlite:///{optuna_dir}/study.db?timeout=300"

    sampler = TPESampler(
        seed=32,
        multivariate=True,
        group=True,
        constant_liar=True,
    )

    study = optuna.create_study(
        study_name="mario_ppo_hyperparameter_tuning",
        direction="maximize",
        sampler=sampler,
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=30,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    study_df = study.trials_dataframe()
    results_path = f"{optuna_dir}/study_results.csv"
    study_df.to_csv(results_path, index=False)

    visualize_study(study, optuna_dir)
