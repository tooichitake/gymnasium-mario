# --- Run best A2C model, print rewards, and record a video ---

import os
import torch as th
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder
# Reuse your env factory (same preprocessing as training)
from mario_cnn_a2c_2 import make_mario_env  # or from mario_a2c_naturecnn import make_mario_env

# ---- Paths (adjust if your layout is different) ----
LOG_ROOT = "results/a2c/exp7"
BEST_DIR = os.path.join(LOG_ROOT, "models", "best_model")
BEST_MODEL = os.path.join(BEST_DIR, "best_model.zip")
VIDEO_DIR = os.path.join(LOG_ROOT, "videos_eval")
os.makedirs(VIDEO_DIR, exist_ok=True)

device = "cuda" if th.cuda.is_available() else "cpu"
model = A2C.load(BEST_MODEL, device=device)
print("Loaded best model:", BEST_MODEL)

# ---- Build evaluation env (VecNormalize on, frozen stats) ----
eval_env = make_mario_env(
    env_id="SuperMarioBros-1-1-v0",
    n_envs=1,
    seed=123,
    frame_stack=4,
    #frame_skip=4,
    #screen_size=84,
    use_vec_normalize=True,
    vec_normalize_kwargs={"training": False, "norm_reward": True},
)

# ---- Print mean/std over 5 episodes ----
mean_r, std_r = evaluate_policy(
    model, eval_env,
    n_eval_episodes=5,
    deterministic=True,
    return_episode_rewards=False
)
print(f"\nMean reward over 5 episodes: {mean_r:.2f} ± {std_r:.2f}")

# ---- Per-episode rewards/lengths (optional) ----
ep_rews, ep_lens = evaluate_policy(
    model, eval_env,
    n_eval_episodes=5,
    deterministic=True,
    return_episode_rewards=True
)
print("\nPer-episode results:")
for i, (r, L) in enumerate(zip(ep_rews, ep_lens), 1):
    print(f"  Episode {i}: reward={r:.2f}, length={L}")

# ---- Make a *video* env (must render frames) ----
# Important: pass render_mode="rgb_array" so VecVideoRecorder can capture frames.
video_env = make_mario_env(
    env_id="SuperMarioBros-1-1-v0",
    n_envs=1,
    seed=456,
    frame_stack=4,
    #frame_skip=4,
    #screen_size=84,
    use_vec_normalize=True,                               # keep eval normalization consistent
    vec_normalize_kwargs={"training": False, "norm_reward": True},
    env_kwargs={"render_mode": "rgb_array"},              # REQUIRED for video recording
)

# Wrap with video recorder: record first episode, up to 10k timesteps
video_env = VecVideoRecorder(
    video_env,
    video_folder=VIDEO_DIR,
    record_video_trigger=lambda step: step == 0,          # record the very first episode
    video_length=18_000,
    name_prefix="mario_A2C_best",
)

# Roll one episode and record
obs = video_env.reset()
done = [False]
episode_reward = 0.0
while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = video_env.step(action)
    episode_reward += reward[0]

video_env.close()
print(f"\n🎥 Saved gameplay video(s) to: {VIDEO_DIR}")
print(f"Recorded episode reward: {episode_reward:.2f}")

#tensorboard --logdir "C:\Users\Tahsin B. Upom\Desktop\Super Mario P3\gymnasium-mario-main\results\a2c_sb3_naturecnn\tensorboard\A2C_2" --port 6006
