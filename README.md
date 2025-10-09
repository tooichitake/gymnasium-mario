# 🎮 Super Mario Bros Reinforcement Learning

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
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

- **🎯 Dual CNN Architectures**: Nature CNN (Mnih et al. 2015) and IMPALA CNN (Espeholt et al. 2018)
- **⚡ Optimized Frame Processing**: Frame skipping with max-pooling and frame stacking
- **🤖 Modern RL Stack**: PPO implementation with Stable-Baselines3
- **🔧 Hyperparameter Tuning**: Optuna integration with AutoSampler
- **📊 Monitoring**: TensorBoard logging and automatic video recording
- **🚄 Parallel Training**: Multiple concurrent environments for faster learning

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- C++ compiler (for NES emulator)
- Conda (recommended)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/tooichitake/gymnasium-mario.git
cd gymnasium-mario

# Create conda environment
conda create -n mario python=3.10
conda activate mario

# Install all dependencies
pip install .
```

## 🚀 Quick Start

### Train with PPO

```python
python mario.py
```

### Hyperparameter Tuning with Optuna

```python
python mario_tune.py
```

### Continue from Checkpoint

```python
python start_from_checkpoint.py
```

## 🔍 CNN-Based Observations

Uses preprocessed image frames with frame skipping and stacking:

```
Observation: 84×84×4 grayscale images
Frame skipping: MaxAndSkipFrame (skip=4) with max-pooling
Frame stacking: VecFrameStack (n_stack=4, consecutive)
Preprocessing: WarpFrame (resize + grayscale)

CNN Architectures:
  1. Nature CNN (Mnih et al. 2015): 3-layer CNN → 512 features
  2. IMPALA CNN (Espeholt et al. 2018): ResNet-style CNN → 256 features (default)
```

### CNN Architecture Comparison

| Feature | Nature CNN | IMPALA CNN |
|---------|-----------|------------|
| **Paper** | Mnih et al. 2015 | Espeholt et al. 2018 |
| **Architecture** | 3 Conv layers | 3 ConvSequences w/ ResBlocks |
| **Residual Connections** | ❌ No | ✅ Yes (2 per sequence) |
| **Max Pooling** | ❌ No | ✅ Yes (3×3, stride=2) |
| **Feature Dimension** | 512 | 256 |
| **Network Depth** | Shallow (3 layers) | Deep (9 layers) |
| **Best For** | Simple tasks | Complex/multi-task |

## 📝 Training Scripts

### `mario.py` - PPO Training

**Features**:
- Visual observations (84×84 grayscale)
- Dual CNN architectures (switchable)
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
- Comparative study of RL algorithms and CNN architectures

GitHub Repository: https://github.com/tooichitake/gymnasium-mario

### Citation
If you use this code in your research, please cite:
```bibtex
@software{zhao2025marioPPO,
  author = {Zhao, Zhiyuan and Yang, Ricki and Rahman, Md Siddiqur},
  title = {Deep Reinforcement Learning for Super Mario Bros: PPO with CNN Architectures},
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
