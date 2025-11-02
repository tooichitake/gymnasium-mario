# tune_a2c_mario_v4.py
import os
import json
import optuna
import numpy as np
import pandas as pd
import torch as th

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# ⬇️ use your project imports as-is
from mario_cnn_a2c_2 import make_mario_env, NatureCNN


# -------------------------
# Paths / study persistence
# -------------------------
ROOT = "results/a2c_tuning_v4"
os.makedirs(ROOT, exist_ok=True)
TB_ROOT = os.path.join(ROOT, "tb")

BEST_JSON = os.path.join(ROOT, "best_params_v4.json")
CSV_OUT   = os.path.join(ROOT, "optuna_trials_v4.csv")

STUDY_NAME = "A2C-Mario-v4"  # brand-new study name
STORAGE_DIR = "results/a2c_tuning_v4"
os.makedirs(STORAGE_DIR, exist_ok=True)
STORAGE = f"sqlite:///{os.path.join(STORAGE_DIR, 'optuna_a2c_v4.sqlite4')}"

# -------------------------------
# Optuna callback with SB3 evaluation
# -------------------------------
class OptunaEvalCallback(EvalCallback):
    def __init__(self, eval_env, trial, n_eval_episodes=3, eval_every_rollouts=20, rollout_size=8192, deterministic=True):
        # translate 'eval_every_rollouts' into eval_freq (timesteps)
        eval_freq = max(1, int(eval_every_rollouts)) * int(rollout_size)
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=None,
            log_path=None,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=False,
            verbose=0,
        )
        self.trial = trial

    def _on_step(self) -> bool:
        ok = super()._on_step()
        if self.last_mean_reward is not None and self.eval_freq > 0:
            # report at evaluation boundaries
            if (self.n_calls % self.eval_freq) == 0:
                self.trial.report(float(self.last_mean_reward), step=int(self.model.num_timesteps))
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
        return ok


# -------------------------
# Policy kwargs factory
# -------------------------
def make_policy_kwargs(trial):
    features_dim = trial.suggest_categorical("features_dim", [256, 512])
    pi_hidden    = trial.suggest_categorical("pi_hidden",    [256, 512, 768])
    vf_hidden    = trial.suggest_categorical("vf_hidden",    [256, 512, 768])

    return dict(
        features_extractor_class=NatureCNN,
        features_extractor_kwargs=dict(features_dim=features_dim, normalized_image=False),
        net_arch=dict(pi=[pi_hidden, pi_hidden], vf=[vf_hidden, vf_hidden]),
    )


# -------------------------
# Objective (fresh v4 space)
# -------------------------
def objective(trial: optuna.trial.Trial):
    # Parallel envs and per-env rollout length (fixed choices → no dynamic spaces)
    n_envs  = int(trial.suggest_categorical("n_envs",  [4, 8]))
    n_steps = int(trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]))

    # Core RL hparams
    gamma         = trial.suggest_float("gamma", 0.97, 0.999, step=0.001)
    gae_lambda    = trial.suggest_float("gae_lambda", 0.90, 0.98, step=0.01)
    ent_coef      = trial.suggest_float("ent_coef", 5e-4, 5e-2, log=True)
    vf_coef       = trial.suggest_float("vf_coef", 0.2, 1.0)
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 0.7, 1.0])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    # Visual pipeline
    frame_skip  = int(trial.suggest_categorical("frame_skip", [2, 4]))
    screen_size = 84  # keep fixed for comparability

    # Optional LR schedule
    use_lr_decay = trial.suggest_categorical("use_lr_decay", [False, True])

    # ------------- Environments -------------
    rollout_size = n_envs * n_steps

    train_env = make_mario_env(
        env_id="SuperMarioBros-1-1-v0",
        n_envs=n_envs,
        seed=0,
        frame_stack=4,
        wrapper_kwargs={"frame_skip": frame_skip, "screen_size": screen_size},
        use_vec_normalize=True,
        vec_normalize_kwargs={"training": True, "norm_reward": True},
        monitor_dir=os.path.join(ROOT, "monitor_train"),
    )

    eval_env = make_mario_env(
        env_id="SuperMarioBros-1-1-v0",
        n_envs=1,
        seed=999,
        frame_stack=4,
        wrapper_kwargs={"frame_skip": frame_skip, "screen_size": screen_size},
        use_vec_normalize=True,
        vec_normalize_kwargs={"training": False, "norm_reward": True},
    )

    # ------------- Model -------------
    policy_kwargs = make_policy_kwargs(trial)
    device = "cuda" if th.cuda.is_available() else "cpu"

    # LR schedule (constant or linear decay to 10% floor)
    lr = float(learning_rate)
    if use_lr_decay:
        base_lr = float(learning_rate)
        def lr_schedule(progress_remaining: float) -> float:
            # progress_remaining: 1.0 → 0.0
            return base_lr * max(0.1, float(progress_remaining))
        lr = lr_schedule

    model = A2C(
        policy="CnnPolicy",
        env=train_env,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        learning_rate=lr,
        tensorboard_log=TB_ROOT,
        policy_kwargs=policy_kwargs,
        device=device,
        seed=0,
        verbose=0,
    )

    # ------------- Training budget -------------
    TOTAL_STEPS = 5_000_000  # 10M per trial

    # Evaluate roughly every ~20 rollouts to reduce overhead
    eval_cb = OptunaEvalCallback(
        eval_env=eval_env,
        trial=trial,
        n_eval_episodes=3,
        eval_every_rollouts=20,
        rollout_size=rollout_size,
        deterministic=True,
    )

    try:
        model.learn(total_timesteps=TOTAL_STEPS, callback=eval_cb, progress_bar=False)
        mean_r, _ = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
    finally:
        train_env.close()
        eval_env.close()

    return float(mean_r)


def main():
    # Fresh study (no previous records)
    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42)
    pruner  = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=40, reduction_factor=3)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        storage=STORAGE,
        load_if_exists=False,   # <- do NOT reuse, start clean
        sampler=sampler,
        pruner=pruner,
    )

    # Configure your trial count here
    study.optimize(objective, n_trials=100, gc_after_trial=True, show_progress_bar=True)

    # Save best params and CSV
    best = {"value": float(study.best_value), "params": study.best_params}
    with open(BEST_JSON, "w") as f:
        json.dump(best, f, indent=2)

    try:
        study.trials_dataframe().to_csv(CSV_OUT, index=False)
    except Exception:
        pass

    print("\n=== Best Trial (v4) ===")
    print("Mean reward:", best["value"])
    print("Params:")
    for k, v in best["params"].items():
        print(f"  {k}: {v}")
    print(f"\nSaved best params JSON: {BEST_JSON}")
    print(f"Saved trials CSV:       {CSV_OUT}")


if __name__ == "__main__":
    main()
