# 🎮 Super Mario Bros Reinforcement Learning

[![Python](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable%20Baselines3-2.7.0-blue.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2.1-brightgreen.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/tooichitake/gymnasium-mario?style=social)](https://github.com/tooichitake/gymnasium-mario/stargazers)

> 🚀 **Train AI agents to play Super Mario Bros using deep reinforcement learning with PPO**

## 🎬 Trained Agent Demo

Watch our PPO agent playing Super Mario Bros after training:

https://github.com/user-attachments/assets/e03ec334-2270-4fac-b583-2a96c2f175cc

*Agent trained with CNN-based observations (video appears 4x speed due to frame_skip=4)*

## 🌟 Key Features

- **IMPALA CNN Architecture**: ResNet-style visual feature extraction
- **Environment Wrappers**: NoopReset, MaxAndSkip (frame skip=4), WarpFrame (84×84 grayscale), VecFrameStack (n=4)
- **Hyperparameter Tuning**: Distributed optimization with automatic checkpoint recovery
- **Training Scripts**: Simple train/test/continue workflow
- **Monitoring**: TensorBoard logging and video recording

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher (3.13+ recommended for better performance)
- C++ compiler (for NES emulator)
- Conda (recommended)

### Quick Install

#### Windows

```bash
# Clone the repository
git clone https://github.com/tooichitake/gymnasium-mario.git
cd gymnasium-mario

# Create conda environment
conda create -n mario python=3.13 -y
conda activate mario

# Install all dependencies
pip install .
```

#### Linux / AWS SageMaker

**First-time setup** (run once):
```bash
# Initialize conda for bash
conda init bash

# Restart shell to apply changes
exec bash
```

**Then install the project**:
```bash
# Clone the repository
git clone https://github.com/tooichitake/gymnasium-mario.git
cd gymnasium-mario

# Create and activate conda environment
conda create -n mario python=3.13 -y
conda activate mario

# Install all dependencies
pip install .
```

**Note**: The project automatically installs `opencv-python-headless` to ensure compatibility with cloud/headless environments.

## 🚀 Quick Start

### Train with PPO

```bash
python train.py
```

Trains on stage 1-1 with IMPALA CNN architecture. Models and logs are saved to `results/ppo/exp{N}/`.

### Test Trained Model

```bash
python test.py
```

Evaluates a trained model and generates gameplay videos in `results/ppo/exp{N}/videos/`.

### Hyperparameter Tuning

```bash
python mario_tune.py
```

Optimizes PPO hyperparameters using Optuna. Results saved to `results/optuna/ppo/`.

### Continue from Checkpoint

```bash
python start_from_checkpoint.py
```

Resumes training from a saved checkpoint. Edit the checkpoint path, training timesteps, and checkpoint frequency directly in the script before running.

## 🔍 CNN-Based Observations

Uses preprocessed image frames with frame skipping and stacking:

```
Observation: 84×84×4 grayscale images
Frame skipping: MaxAndSkipFrame (skip=4) with max-pooling
Frame stacking: VecFrameStack (n_stack=4, consecutive)
Preprocessing: WarpFrame (resize + grayscale)
Noop Reset: NoopResetEnv (1-80 random no-ops on reset)

CNN Architecture:
  IMPALA CNN (Espeholt et al. 2018): ResNet-style CNN → 256 features
```

**Preprocessing Pipeline**:
1. **NoopResetEnv**: Execute 1-80 random no-op actions on reset (adds starting position variety)
2. **MaxAndSkipFrame**: Skip 4 frames, return max-pooled frame (reduces temporal redundancy)
3. **WarpFrame**: Resize to 84×84 and convert to grayscale (standard Atari preprocessing)
4. **VecFrameStack**: Stack 4 consecutive frames (provides motion information)

### IMPALA CNN Architecture

**Paper**: Espeholt, L., et al. (2018). *IMPALA: Scalable Distributed Deep-RL.* ICML 2018.

| Feature | Details |
|---------|---------|
| **Architecture** | 3 ConvSequences with ResBlocks |
| **Residual Connections** | ✅ Yes (2 per sequence) |
| **Max Pooling** | ✅ Yes (3×3, stride=2) |
| **Feature Dimension** | 256 |
| **Network Depth** | Deep (9 layers: 3 conv + 6 residual) |
| **Best For** | Complex/multi-task learning |

## 📝 Training Scripts

### `train.py` - PPO Training

**Features**:
- Visual observations (84×84 grayscale)
- IMPALA CNN architecture
- CnnPolicy with frame stacking
- Automatic checkpoint saving

**Configuration**:
```python
frame_skip = 4     # Frame skipping with max-pooling
screen_size = 84   # Resize dimension
n_envs = 4         # Parallel environments
total_timesteps = 2_000_000
checkpoint_freq = 50_000
```

### `test.py` - Model Evaluation

**Features**:
- Load trained models from checkpoints
- Visual observations (84×84 grayscale)
- Automatic video recording
- Episode statistics (CSV output)

**Configuration**:
```python
checkpoint_path = "results/ppo/exp1/models/checkpoints/mario_PPO_2000000_steps.zip"
n_episodes = 5     # Number of test episodes
deterministic = True
```

### `start_from_checkpoint.py` - Continue Training

**Features**:
- Resume training from saved checkpoint
- Maintains training environment configuration
- Continues checkpoint saving
- Preserves timestep count

**Configuration**:
```python
checkpoint_path = "results/ppo/exp1/models/checkpoints/mario_PPO_1000000_steps.zip"
total_timesteps = 1_000_000   # Additional training steps
checkpoint_freq = 50_000       # Checkpoint save frequency
n_envs = 4                     # Match original training
```

### `mario.py` - PPO Training (Legacy)

**Features**:
- Visual observations (84×84 grayscale)
- IMPALA CNN architecture
- CnnPolicy with frame stacking
- Automatic video recording

**Configuration**:
```python
frame_skip = 4     # Frame skipping with max-pooling
screen_size = 84   # Resize dimension
frame_stack = 4    # Number of frames to stack
n_envs = 8         # Parallel environments
total_timesteps = 15_000_000
```

### `mario_tune.py` - Hyperparameter Tuning

**Features**:
- Optuna integration with AutoSampler
- Continuous variable optimization
- Persistent storage (SQLite)
- Automatic visualization generation

**Hyperparameters Tuned**:
```python
n_steps: [1024, 2048, 4096]
batch_size: [64, 128, 256, 512, 1024, 2048, 4096]
n_epochs: 4-16
learning_rate: 1e-5 to 1e-3 (log scale)
gamma: 0.90-0.999
gae_lambda: 0.90-0.999
ent_coef: 0.001-0.05 (log scale)
clip_range: 0.1-0.5
vf_coef: 0.3-1.0
max_grad_norm: 0.3-1.0
```

## 📊 Results & Monitoring

### Training Output Structure

```
results/
├── ppo/
│   └── exp1/
│       ├── models/           # Model checkpoints
│       ├── logs/             # TensorBoard logs
│       └── videos/           # Gameplay recordings
└── optuna/
    └── ppo/
        ├── study.db          # Optuna database
        ├── best_model.zip    # Best model from tuning
        └── *.html            # Visualization plots
```

### TensorBoard Monitoring

```bash
tensorboard --logdir results/ppo/exp1/logs
```

Metrics include:
- Episode rewards and lengths
- Policy/value losses
- Learning rate schedule
- Frames per second (FPS)

## ⚙️ Advanced Configuration

### Wrapper Parameters

The `make_mario_env` function accepts several wrapper parameters for customizing environment behavior:

| Parameter | Default | Description |
|-----------|---------|-------------|
| **frame_skip** | 4 | Number of frames to skip (action repeated). Higher = faster training but less control. |
| **screen_size** | 84 | Resize frame to square image (84×84). Standard for Atari-style RL. |
| **noop_max** | 80 | Maximum random no-op actions on reset. Adds stochasticity to starting positions. Set to 0 to disable. |
| **use_single_stage_episodes** | False | If True, episode ends after completing one stage (flag captured). If False, continues to next stage (1-1 → 1-2 → ...). |

**Example usage**:
```python
# Training: Single-stage episodes with high noop randomization
train_env = make_mario_env(
    "SuperMarioBros-1-1-v0",
    wrapper_kwargs={
        "frame_skip": 4,              # Skip 4 frames per action
        "screen_size": 84,             # 84×84 image
        "noop_max": 80,                # 1-80 random no-ops on reset
        "use_single_stage_episodes": True,  # End after stage completion
    }
)

# Testing: Disable noop for deterministic evaluation
test_env = make_mario_env(
    "SuperMarioBros-1-1-v0",
    wrapper_kwargs={
        "frame_skip": 4,
        "screen_size": 84,
        "noop_max": 0,                 # No random no-ops
        "use_single_stage_episodes": True,
    }
)
```

### Training Stages

**Current implementation** (multi-stage training):

```python
# Training: All 8 stages (World 1-2)
train_env = make_mario_env(
    ["SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0", ..., "SuperMarioBros-2-4-v0"],
    n_envs=8,
    ...
)

# Evaluation: Fixed stage 1-1
eval_env = make_mario_env(
    "SuperMarioBros-1-1-v0",
    n_envs=1,
    ...
)
```

### Reward Function

The reward function is: **r = v + c + d** (clipped to [-15, 15])

| Component | Description | Value |
|-----------|-------------|-------|
| **v** (Velocity) | Horizontal movement | `x1 - x0` |
| **c** (Clock) | Time penalty | `c0 - c1` |
| **d** (Death) | Death penalty | -15 |

### Reward Normalization

Uses VecNormalize for reward normalization:
- Normalizes by standard deviation of discounted returns
- Gamma = 0.99
- Clip range = [-10, 10]

## 👏 Credits

### Authors
- **Zhiyuan Zhao**
- **Ricki Yang**
- **Md Siddiqur Rahman**

**Key Contributions**:
- PyTorch implementation of IMPALA CNN for Stable-Baselines3
- Gymnasium-compatible fork of `nes-py` and `gymnasium-super-mario-bros`
- Hyperparameter optimization with Optuna AutoSampler
- PPO training with IMPALA CNN architecture

GitHub Repository: https://github.com/tooichitake/gymnasium-mario

### Citation
If you use this code in your research, please cite:
```bibtex
@software{zhao2025marioPPO,
  author = {Zhao, Zhiyuan and Yang, Ricki and Rahman, Md Siddiqur},
  title = {Deep Reinforcement Learning for Autonomous Super Mario Bros Gameplay: A Comparative Study of RL Algorithms},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/tooichitake/gymnasium-mario}
}
```

### Acknowledgments
- **[Christian Kauten](https://github.com/Kautenja)** - Original `nes-py` and `gym-super-mario-bros`
- **[OpenAI](https://github.com/openai)** - Original Gym framework

### Built With
- [Gymnasium](https://gymnasium.farama.org/) - RL environment API
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [Optuna](https://optuna.org/) - Hyperparameter optimization
- [PyTorch](https://pytorch.org/) - Deep learning framework

### References

**[1]** Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518(7540):529-533. [[Paper]](https://www.nature.com/articles/nature14236)

**[2]** Espeholt, L., et al. (2018). *IMPALA: Scalable Distributed Deep-RL.* ICML 2018. [[Paper]](https://arxiv.org/abs/1802.01561)

**[3]** Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* [[Paper]](https://arxiv.org/abs/1707.06347)

**[4]** Engstrom, L., et al. (2020). *Implementation Matters in Deep Policy Gradients.* ICLR 2020. [[Paper]](https://arxiv.org/abs/2005.12729)

## 📄 License

MIT License - see LICENSE file for details

---

<p align="center">
  Made with ❤️ for the RL community
</p>
