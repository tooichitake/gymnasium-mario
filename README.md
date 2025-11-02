# 🎮 Super Mario Bros Reinforcement Learning

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable%20Baselines3-2.7.0-blue.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2.1-brightgreen.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/tooichitake/gymnasium-mario?style=social)](https://github.com/tooichitake/gymnasium-mario/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/tooichitake/gymnasium-mario?style=social)](https://github.com/tooichitake/gymnasium-mario/network/members)

> 🚀 **Train AI agents to play Super Mario Bros using both grid-based (RAM) and CNN-based (visual) approaches**

## 🎬 Trained Agent Demo

Watch our A2C agent playing Super Mario Bros after training:

https://github.com/user-attachments/assets/c2bb0f7c-6ccb-460c-95b9-df79760814e7

*Agent trained with CNN-based observations (video appears 4x speed due to frame_skip=4)*

## 🌟 Key Features

- **🎯 Dual Observation Methods**:
  - Grid-based: 13×16 semantic grid from RAM (ultra-fast, interpretable)
  - CNN-based: Image observations with frame skipping (standard RL approach)
- **⚡ Optimized Frame Stacking**:
  - Grid: Interval-based stacking with circular buffer
  - CNN: Consecutive stacking after frame skip
- **🎮 Configurable Frame Skip**: Adjustable `frame_skip` parameter (default=4)
- **🤖 Modern RL Stack**: PPO implementation with Stable-Baselines3
- **📊 Video Recording**: Automatic gameplay recording with VecVideoRecorder
- **🚄 Parallel Training**: Multiple concurrent environments for faster learning

## 📋 Table of Contents

- [Overview](#️-overview)
- [Installation](#️-installation)
- [Quick Start](#-quick-start)
- [Observation Methods](#-observation-methods)
- [Training Scripts](#-training-scripts)
- [Results & Videos](#-results--videos)
- [Project Structure](#-project-structure)
- [Advanced Configuration](#️-advanced-configuration)
- [Credits](#-credits)

## 🎯 Overview

This repository provides two complementary approaches for training Mario agents:

### 1. Grid-Based (RAM) Approach
- **Fast**: 10x faster than CNN-based methods
- **Interpretable**: Clear semantic meaning (Mario=2, Enemy=1, Platform=0)
- **Memory efficient**: 1.6KB vs 320KB per observation
- **CPU-friendly**: No GPU required

### 2. CNN-Based (Visual) Approach
- **Standard RL**: Industry-standard visual observations
- **Dual CNN architectures**: Nature CNN (Mnih et al. 2015) and IMPALA CNN (Espeholt et al. 2018)
- **Frame stacking**: Temporal information with circular buffer
- **Normal speed**: Game runs at 60 FPS (not accelerated)
- **Watchable videos**: Recorded gameplay is smooth and viewable

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- C++ compiler (for NES emulator)
- Conda (recommended)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/SiddiqurRahman3802/gymnasium-mario.git
cd gymnasium-mario

# Create conda environment
conda create -n mario python=3.12
conda activate mario

# Install all dependencies (including NES environment)
pip install .
```

## 🚀 Quick Start

### Train with Grid-Based Observations (Fast)

```python
python mario_grid.py
```

### Train with CNN-Based Observations (Standard)

```python
python mario_cnn.py
```

### Continue from Checkpoint

```python
python start_from_checkpoint.py
```

## 🔍 Observation Methods

### Grid-Based Observations (`mario_grid.py`)

Converts game state into a **13×16 semantic grid** from NES RAM:

```
Grid Values:
  -1 = Empty space/background
   0 = Platforms/pipes (neutral objects)
   1 = Enemies (Goombas, Koopas, etc.)
   2 = Mario (player position)
```

**Advantages**:
- Ultra-fast training (10x faster)
- Interpretable states
- Small observation space (13×16×n_stack)
- Works well with MLP policy

### CNN-Based Observations (`mario_cnn.py`)

Uses preprocessed image frames with frame skipping and VecFrameStack:

```
Observation: 84×84×4 grayscale images
Frame skipping: MaxAndSkipFrame (skip=4, configurable) with max-pooling
Frame stacking: VecFrameStack (n_stack=4, configurable, consecutive)
Preprocessing: WarpFrame (resize + grayscale)

CNN Architectures (switchable):
  1. Nature CNN (Mnih et al. 2015): 3-layer CNN → 3136 features → 512
  2. IMPALA CNN (Espeholt et al. 2018): ResNet-style CNN → 3872 features → 256
```

**Frame Skipping Mechanism** (`MaxAndSkipFrame`):
- **Action repeat**: Same action executes for `skip=4` consecutive frames
- **Observation sampling**: Only returns max-pooled frame from last 2 frames
- **Skipped frames**: Intermediate 3 frames are **not sampled or recorded**
- **Effective game speed**: Agent observes game at 1/4 speed (15 decisions/sec vs 60 FPS)
- **Video recording**: Only agent decision frames are captured → **videos appear 4x speed**

**Frame Stacking** (`VecFrameStack`):
- Stacks `n_stack=4` **consecutive sampled frames** (after frame skipping)
- Provides temporal information from last 4 agent decisions
- Total temporal coverage: 16 game frames (4 stacked × 4 skip)

**Key Features**:
- Frame skipping reduces computation by 4x
- Max-pooling handles NES sprite flickering
- Both `frame_skip` and `frame_stack` are configurable parameters
- VecTransposeImage prepares for CNN input (channels-first format)

**Important Notes**:
- ⚠️ **Recorded videos are 4x speed** due to frame skipping
- Game runs at normal 60 FPS internally, but agent only sees every 4th frame
- To get real-time videos, set `frame_skip=1` (slower training)

## 📝 Training Scripts

### `mario_grid.py` - Grid-Based Training

**Features**:
- RAM-based observations
- MLP policy (256×256 hidden layers)
- Fast training (CPU-friendly)
- Custom `MarioWrapper` for RAM extraction

**Configuration**:
```python
n_stack = 4        # Number of frames to stack
n_skip = 4         # Stacking interval (skip frames)
n_envs = 1         # Parallel environments
total_timesteps = 30_000_000
```

### `mario_cnn.py` - CNN-Based Training

**Features**:
- Visual observations (84×84 grayscale)
- **Dual CNN architectures**: Nature CNN or IMPALA CNN (configurable)
- CnnPolicy with frame stacking
- Normal game speed (no acceleration)
- Automatic video recording

**CNN Architecture Comparison**:

| Feature | Nature CNN | IMPALA CNN |
|---------|-----------|------------|
| **Paper** | Mnih et al. 2015 | Espeholt et al. 2018 |
| **Architecture** | 3 Conv layers | 3 ConvSequences w/ ResBlocks |
| **Residual Connections** | ❌ No | ✅ Yes (2 per sequence) |
| **Max Pooling** | ❌ No | ✅ Yes (3×3, stride=2) |
| **Feature Dimension** | 3136 → 512 | 3872 → 256 |
| **Compression Ratio** | 6.1× | 15.1× |
| **Network Depth** | Shallow (3 layers) | Deep (3 + 6 residual) |
| **Best For** | Simple/single-task | Complex/multi-task |
| **Training Speed** | Faster | Slightly slower |
| **Generalization** | Good | Better (ResNet benefits) |

**Preprocessing Pipeline**:
```python
# MarioWrapper applies:
env = JoypadSpace(env, SIMPLE_MOVEMENT)      # 7 actions
env = MaxAndSkipFrame(env, skip=4)           # Frame skip + max pool
env = WarpFrame(env, width=84, height=84)    # Resize + grayscale

# Then wrapped with:
VecFrameStack(vec_env, n_stack=4)            # Stack 4 consecutive frames
VecTransposeImage(vec_env)                   # CHW format for CNN
```

**Configuration**:
```python
frame_skip = 4     # Frame skipping with max-pooling
screen_size = 84   # Resize dimension
frame_stack = 4    # VecFrameStack frames
n_envs = 8         # Parallel environments
total_timesteps = 10_000_000
```

### `start_from_checkpoint.py` - Resume Training

Resume training from a saved model checkpoint:

```python
# Load checkpoint
model = PPO.load("results/ppo/exp1/models/best_model/best_model.zip")

# Continue training
model.learn(total_timesteps=10_000_000)
```

## 📊 Results & Videos

### Training Output Structure

```
results/
└── ppo/
    └── exp1/
        ├── models/
        │   ├── best_model/
        │   │   └── best_model.zip
        │   ├── rl_model_500000_steps.zip
        │   └── rl_model_1000000_steps.zip
        ├── logs/
        │   └── (TensorBoard logs)
        └── videos/
            ├── mario_PPO_gameplay-step-0-to-step-10000.mp4
            └── mario_PPO_gameplay-step-10001-to-step-20001.mp4
```

### Viewing Videos

Videos are automatically recorded during training using `VecVideoRecorder`:

**CNN-based videos** (`mario_cnn.py`):
- Appear **4x faster** than real-time due to `frame_skip=4`
- Only frames where agent makes decisions are recorded (every 4th game frame)
- Smooth but accelerated gameplay
- To record at normal speed: set `frame_skip=1` in MarioWrapper (slower training)

**Grid-based videos** (`mario_grid.py`):
- Show agent decision frequency based on `n_skip` parameter
- No frame skipping in base environment

### TensorBoard Monitoring

```bash
tensorboard --logdir results/ppo/exp1/logs
```

Metrics include:
- Episode rewards and lengths
- Policy/value losses
- Learning rate schedule
- Frame-per-second (FPS)

## 📁 Project Structure

```
gymnasium-mario-master/
├── 📄 mario_grid.py              # Grid-based (RAM) training
├── 📄 mario_cnn.py               # CNN-based (visual) training
├── 📄 start_from_checkpoint.py  # Resume training script
├── 📄 slow_down_videos.py       # Video post-processing (deprecated)
├── 📄 README.md                  # This file
├── 📊 results/                   # Training outputs
│   └── ppo/
│       └── exp1/
│           ├── models/          # Saved checkpoints
│           ├── logs/            # TensorBoard logs
│           └── videos/          # Gameplay recordings
└── 🔧 dependencies/              # Local NES environment packages
```

## ⚙️ Advanced Configuration

### Random Stage Training (Improves Generalization)

Train on multiple stages to improve generalization and robustness:

**Current implementation in `mario_cnn.py`** (default):

```python
# Training: All 32 stages (maximum generalization)
train_env = make_mario_env(
    "SuperMarioBrosRandomStages-v0",
    n_envs=8,
    seed=0,
    ...
)

# Evaluation: Fixed stage 1-1 (consistent benchmarking)
eval_env = make_mario_env(
    "SuperMarioBros-1-1-v0",
    n_envs=1,
    seed=1,
    ...
)
```

**How Random Stage Selection Works**:

1. **Per-environment randomization**: With `n_envs=8` and `seed=0`, each parallel environment gets a different seed (0, 1, 2, ..., 7) via `make_vec_env`
2. **Initial stage diversity**: At first reset, 8 environments load 8 different random stages simultaneously
3. **Continuous randomization**: Every episode reset selects a new random stage using the environment's RNG
4. **Example training sequence**:
   ```
   Episode 1: Env0→5-4, Env1→6-4, Env2→1-4, Env3→3-1, ... (8 different stages)
     ↓ Reset after completion/failure
   Episode 2: Env0→6-1, Env1→2-3, Env2→7-2, Env3→1-1, ... (8 new random stages)
     ↓ Reset
   Episode 3: Env0→4-4, Env1→1-1, Env2→3-4, Env3→8-2, ... (continues randomly)
   ```

**Alternative Configurations**:

```python
# Option A: Subset of stages (e.g., World 1 only)
train_env = make_mario_env(
    "SuperMarioBrosRandomStages-v0",
    n_envs=8,
    env_kwargs={"stages": ['1-1', '1-2', '1-3', '1-4']},
    ...
)

# Option B: Single stage training (for comparison)
train_env = make_mario_env(
    "SuperMarioBros-1-1-v0",
    n_envs=8,
    ...
)
```

**Benefits**:
- ✅ **Improved generalization**: Agent learns robust features, not stage-specific patterns
- ✅ **Better transfer**: Can play unseen stages more effectively
- ✅ **Reduced overfitting**: Less memorization of single-stage quirks
- ✅ **Maximum diversity**: 8 parallel envs × continuous random stages = high sample diversity
- ✅ **Addresses frame skip dependency**: Diverse stages require adaptable timing

**Trade-offs**:
- ⚠️ Slower convergence (more diverse training data)
- ⚠️ May need more training steps (20-30M instead of 10M)
- ⚠️ Harder to master any single stage perfectly

**Deterministic Evaluation**:
- Evaluation uses `deterministic=True` + fixed seed → same actions every time
- Only `n_eval_episodes=1` needed (no variance in deterministic environments)
- Consistent benchmarking on stage 1-1 to track progress

### Episode Behavior and Stage Completion

**Important: Both `SuperMarioBros-1-1-v0` and `SuperMarioBrosRandomStages-v0` allow multi-stage episodes**

#### Episode Termination Conditions

Episodes end **only** when:
1. ✅ **Death**: Mario dies (falls in pit, hits enemy, time runs out)
2. ✅ **Game Over**: All lives lost (life counter reaches 0)
3. ❌ **NOT on stage completion**: Completing a stage does NOT trigger `done=True`

#### Stage Completion Behavior

```python
# Both environments behave the same way:
# SuperMarioBros-1-1-v0
Episode starts → 1-1 → Complete → 1-2 → Complete → 1-3 → Death → Episode ends
                                  ↑ Timer resets to 400, episode continues

# SuperMarioBrosRandomStages-v0
Reset → Random 5-2 → Complete → 5-3 → Complete → 5-4 → Death → Episode ends
         ↑ Initial random selection    ↑ Timer resets, episode continues
```

**Key Points**:
- ✅ Completing a stage **continues the episode** to the next stage
- ✅ Game timer **resets to 400** on each new stage
- ✅ Episodes can span multiple stages (even multiple worlds)
- ✅ `SuperMarioBros-1-1-v0`: Always starts at 1-1 on reset, but progresses naturally
- ✅ `SuperMarioBrosRandomStages-v0`: Randomly selects starting stage on reset, then progresses naturally

#### Episode Length and `max_episode_steps`

**Default behavior** (when `max_episode_steps=None`):
- Uses environment default: **9,999,999 steps**
- Essentially unlimited for practical purposes
- Game's internal timer provides natural episode limits (~8019 steps per stage if idle)

**Why we don't limit `max_episode_steps`**:
- ✅ **Game timer is sufficient**: Each stage has 400 time units (≈8000 steps), automatically ends on timeout
- ✅ **Multi-stage episodes**: Excellent agents can complete multiple stages (1-1→1-2→1-3→1-4), requiring 30,000+ steps
- ✅ **No infinite loops**: Game mechanics prevent infinite episodes (timer + death conditions)
- ⚠️ **Setting too low** (e.g., 8019) truncates episodes prematurely, preventing multi-stage achievements

**Example episode lengths**:
```
Poor agent (stands still):        8,019 steps (timeout death on first stage)
Average agent (dies in 1-2):      ~15,000 steps (completes 1-1, dies in 1-2)
Good agent (completes World 1):   ~35,000 steps (1-1→1-2→1-3→1-4, dies in 2-1)
Excellent agent:                  50,000+ steps (multiple worlds)
```

**Recommendation**: Use default `max_episode_steps=None` to allow agents to demonstrate multi-stage capabilities.

### Frame Stacking vs Frame Skipping

**Grid-based approach (`n_skip` parameter in MarioWrapper)**:
- `n_skip=4`: Interval-based stacking [current, -4, -8, -12] frames
- Captures motion over longer time periods
- More efficient temporal coverage
- Used only in `mario_grid.py`

**CNN-based approach (frame_skip in MaxAndSkipFrame)**:
- `frame_skip=4`: Action repeats 4 times, returns max-pooled frame
- Reduces computation by 4x
- Handles NES sprite flickering
- Then VecFrameStack adds 4 consecutive processed frames
- Used in `mario_cnn.py`

### Environment Wrappers

#### Grid-based (mario_grid.py)
```python
env = JoypadSpace(env, SIMPLE_MOVEMENT)      # 7 actions
env = MarioWrapper(env, n_stack=4, n_skip=4) # RAM grid + stacking
```

#### CNN-based (mario_cnn.py)
```python
env = JoypadSpace(env, SIMPLE_MOVEMENT)      # 7 actions
env = MaxAndSkipFrame(env, skip=4)           # Frame skip + max pool
env = WarpFrame(env, width=84, height=84)    # Resize + grayscale
# Then: VecFrameStack + VecTransposeImage
```

### Reward Function

The reward function is: **r = v + c + d** (clipped to range [-15, 15])

| Component | Description | Value | Meaning |
|-----------|-------------|-------|---------|
| **v** (Velocity) | Horizontal movement | `v = x1 - x0` | Difference in x position |
| | Moving right | v > 0 | Positive reward |
| | Moving left | v < 0 | Negative penalty |
| | Standing still | v = 0 | No reward |
| **c** (Clock) | Time penalty | `c = c0 - c1` | Difference in game clock |
| | Clock tick | c < 0 | Small penalty |
| | No clock tick | c = 0 | No penalty |
| **d** (Death) | Death penalty | d = -15 | When Mario dies |
| | Alive | d = 0 | When Mario is alive |

**Goal**: Move right as fast as possible without dying

### Reward Normalization

Both training scripts use **VecNormalize** for reward normalization to stabilize training:

**Algorithm** (from Stable-Baselines3 source code):

1. **Update discounted returns**:
   ```
   R_t = γ·R_{t-1} + r_t
   ```
   where `R_t` is the running return, `γ=0.99` is discount factor, `r_t` is current reward

2. **Update running statistics** (Welford's online algorithm):
   ```
   δ = μ_batch - μ_old
   n_total = n_old + n_batch

   μ_new = μ_old + δ · (n_batch / n_total)

   M2_new = M2_old + M2_batch + δ² · (n_old · n_batch / n_total)
   σ²_new = M2_new / n_total
   ```
   where `μ` is mean, `σ²` is variance, `n` is count, `M2` is sum of squared differences

3. **Normalize rewards**:
   ```
   r̃_t = clip(r_t / √(σ²_R + ε), -c, c)
   ```
   where `σ²_R` is variance of returns, `ε=1e-8` prevents division by zero, `c=10` is clip range

**Implementation Details**:
- **Source**: [`stable_baselines3/common/vec_env/vec_normalize.py`](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py)
- **Running statistics**: Uses parallel variance algorithm from [Wikipedia](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm)
- **Key insight**: Normalizes by **standard deviation of discounted returns**, not raw rewards
- **Episode handling**: Returns reset to 0 when episode ends

**Parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `norm_reward` | `True` | Enable reward normalization |
| `norm_obs` | `False` | Don't normalize observations (already preprocessed) |
| `gamma` | `0.99` | Discount factor for return calculation |
| `clip_reward` | `10.0` | Clip normalized rewards to [-10, 10] |
| `epsilon` | `1e-8` | Small constant to avoid division by zero |

**Benefits**:
- Stabilizes learning across different reward scales
- Prevents large reward spikes from destabilizing training
- Improves convergence speed and sample efficiency
- Essential for PPO performance on continuous/variable-reward tasks

**Reference**: Engstrom et al., "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO" (ICLR 2020) [arXiv:2005.12729](https://arxiv.org/abs/2005.12729)

### PPO Hyperparameters

#### Grid-based (mario_grid.py)
```python
learning_rate = LinearSchedule(4e-4, 2e-4)
n_steps = 8192
batch_size = 2048
n_epochs = 10
gae_lambda = 0.95
clip_range = LinearSchedule(0.22, 0.18)
ent_coef = 0.01
vf_coef = 0.5
policy = MlpPolicy (256×256)
```

#### CNN-based (mario_cnn.py)
```python
learning_rate = 3e-4
n_steps = 4096
batch_size = 8192
n_epochs = 10
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.005
vf_coef = 0.5

# Architecture options (switchable in mario_cnn.py):
# Option 1: Nature CNN (Mnih et al. 2015)
policy_kwargs = dict(
    features_extractor_class=NatureCNN,
    features_extractor_kwargs=dict(features_dim=512, normalized_image=False),
    net_arch=dict(pi=[512, 512], vf=[512, 512]),
)

# Option 2: IMPALA CNN (Espeholt et al. 2018) - Current default
policy_kwargs = dict(
    features_extractor_class=ImpalaCNN,
    features_extractor_kwargs=dict(
        features_dim=256,
        depths=[16, 32, 32],  # Channel depths for each ConvSequence
        normalized_image=False
    ),
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
)
```

## 🔧 Bug Fixes & Improvements

### Recent Bug Fixes & Improvements

1. **Grid-based Frame Stacking**:
   - **Implementation**: Circular buffer with interval-based stacking (n_skip=4)
   - **Benefit**: Efficient temporal coverage [current, -4, -8, -12] frames
   - **Impact**: Captures motion over longer time periods

2. **CNN-based Preprocessing**:
   - **Frame skipping**: MaxAndSkipFrame with max-pooling to handle sprite flickering
   - **Frame stacking**: VecFrameStack for consecutive 4 frames
   - **Optimization**: Proper image transposing for CNN policy

3. **Training Improvements**:
   - VecNormalize for reward normalization
   - Linear learning rate schedules (grid-based)
   - Parallel environments (8 envs for CNN, 1 for grid)
   - Checkpoint and evaluation callbacks

4. **Neural Network Initialization** (New):
   - **Orthogonal Weight Initialization**: Applied to both NatureCNN and ImpalaCNN feature extractors
   - **Layer-specific Scaling**: CNN/Linear layers use gain=√2 for ReLU activation
   - **Improved Gradient Flow**: Better weight initialization enables deeper networks to train effectively
   - **Implementation**: Custom `_initialize_weights()` method in both CNN classes
   - **Activation Functions**:
     - Feature extractors (CNN): ReLU activation
     - Policy/Value MLPs: Tanh activation (Stable-Baselines3 default)
   - **Benefits**: Faster convergence, better performance, reduced gradient vanishing
   - **References**: See [4], [6], [7] in References section below

5. **Normalization Strategy**:
   - **Reward Normalization**: Enabled (`norm_reward=True`) for stable training across varying reward scales
   - **Observation Normalization**: Disabled (`norm_obs=False`) for image/grid observations
   - **Rationale**:
     - Images are already normalized via `/255.0` preprocessing
     - Grid observations are discrete semantic values (-1, 0, 1, 2)
     - Observation normalization is primarily beneficial for continuous state spaces (e.g., MuJoCo)
   - **Note**: This follows best practices for visual RL while adapting the "Implementation Matters" recommendations
   - **Reference**: See [4] in References section below

### Code Quality

- ✅ Type hints for all functions
- ✅ Comprehensive docstrings
- ✅ Modular design (wrappers, environments, training)
- ✅ Efficient circular buffer implementation

## 🤝 Contributing

Contributions welcome! Areas of interest:

- [ ] Additional observation representations
- [ ] Curriculum learning experiments
- [ ] Multi-stage training
- [ ] Hyperparameter optimization
- [ ] Web-based visualization

## 👏 Credits

### Authors
This project was developed by:
- **Zhiyuan Zhao**
- **Ricki Yang**
- **Md Siddiqur Rahman**

**Key Contributions**:
- PyTorch implementation of IMPALA CNN architecture for Stable-Baselines3
- Gymnasium-compatible versions of `nes-py` and `gymnasium-super-mario-bros` (forked and adapted)
- Dual observation approaches: Grid-based (RAM) and CNN-based (Visual)
- Comparative study of reinforcement learning algorithms and architectures

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
- **[Christian Kauten](https://github.com/Kautenja)** - Original author of `nes-py` and `gym-super-mario-bros`
  - We forked and adapted these packages for Gymnasium compatibility (included in this repository)
- **[OpenAI](https://github.com/openai)** - Original Gym framework

### Built With
- [Gymnasium](https://gymnasium.farama.org/) - Modern RL environment API
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Image processing
- [MoviePy](https://zulko.github.io/moviepy/) - Video processing

### References

**[1]** Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis.
*Human-level control through deep reinforcement learning.*
**Nature**, 518(7540):529-533, 2015.
[[Paper]](https://www.nature.com/articles/nature14236) [[DOI]](https://doi.org/10.1038/nature14236)

**[2]** Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, and Koray Kavukcuoglu.
*IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures.*
In **Proceedings of the 35th International Conference on Machine Learning (ICML)**, volume 80, pages 1407-1416. PMLR, 2018.
[[Paper]](https://arxiv.org/abs/1802.01561) [[PDF]](https://arxiv.org/pdf/1802.01561.pdf)

**[3]** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
*Proximal Policy Optimization Algorithms.*
arXiv preprint arXiv:1707.06347, 2017.
[[Paper]](https://arxiv.org/abs/1707.06347) [[PDF]](https://arxiv.org/pdf/1707.06347.pdf)

**[4]** Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, and Aleksander Madry.
*Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO.*
In **International Conference on Learning Representations (ICLR)**, 2020.
[[Paper]](https://arxiv.org/abs/2005.12729) [[PDF]](https://arxiv.org/pdf/2005.12729.pdf) [[OpenReview]](https://openreview.net/forum?id=r1etN1rtPB)

**[5]** Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
*Deep Residual Learning for Image Recognition.*
In **Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)**, pages 770-778, 2016.
[[Paper]](https://arxiv.org/abs/1512.03385) [[PDF]](https://arxiv.org/pdf/1512.03385.pdf) [[DOI]](https://doi.org/10.1109/CVPR.2016.90)

**[6]** Andrew M. Saxe, James L. McClelland, and Surya Ganguli.
*Exact solutions to the nonlinear dynamics of learning in deep linear neural networks.*
In **International Conference on Learning Representations (ICLR)**, 2014.
[[Paper]](https://arxiv.org/abs/1312.6120) [[PDF]](https://arxiv.org/pdf/1312.6120.pdf)

**[7]** Stable-Baselines3 Team.
*Custom Feature Extractors Documentation.*
Stable-Baselines3 Documentation, 2024.
[[Docs]](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
*Note: Our IMPALA CNN implementation is available in `mario_cnn.py` (lines 111-241).*

## 📄 License

MIT License - see LICENSE file for details

---

<p align="center">
  Made with ❤️ for the RL community
</p>
