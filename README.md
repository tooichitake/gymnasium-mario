# Mario RL Training with Grid-Based Observations

[![Test](https://github.com/tooichitake/gymnasium-mario/actions/workflows/test.yml/badge.svg)](https://github.com/tooichitake/gymnasium-mario/actions/workflows/test.yml)
[![Lint](https://github.com/tooichitake/gymnasium-mario/actions/workflows/lint.yml/badge.svg)](https://github.com/tooichitake/gymnasium-mario/actions/workflows/lint.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-CPU-orange.svg)](https://github.com/google/jax)
[![Stable-Baselines3](https://img.shields.io/badge/Stable%20Baselines3-2.7.0-blue.svg)](https://stable-baselines3.readthedocs.io/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-brightgreen.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-Compatible-blue.svg)](https://stable-baselines3.readthedocs.io/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![GitHub stars](https://img.shields.io/github/stars/tooichitake/gymnasium-mario.svg)](https://github.com/tooichitake/gymnasium-mario/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/tooichitake/gymnasium-mario.svg)](https://github.com/tooichitake/gymnasium-mario/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/tooichitake/gymnasium-mario.svg)](https://github.com/tooichitake/gymnasium-mario/commits/main)

This repository contains implementations for training Super Mario Bros agents using reinforcement learning, featuring both pixel-based and innovative grid-based observation approaches.

## Overview

This project includes:
- **Grid-based Mario environment**: A novel approach using RAM-extracted grid representations
- **Pixel-based Mario environment**: Traditional CNN-based approach
- **Gymnasium-compatible ports**: Modern versions of `nes-py` and `gymnasium-super-mario-bros`
- **Stable-Baselines3 integration**: State-of-the-art PPO implementation with JAX support

## Features

### 🎮 Grid-Based Approach (`mario_grid_jax.py`)
- Extracts game state directly from NES RAM
- Converts to 13x16 grid representation
- Significantly faster training than pixel-based methods
- Uses MLP policy instead of CNN
- Memory efficient (only 1.6KB per observation)

### 📺 Pixel-Based Approach (`mario_pixel.py`) 
- Traditional 84x84 grayscale frames
- Frame stacking and skipping
- CNN policy for visual processing

## Project Structure

```
gymnasium-mario-master/
├── mario_grid_jax.py           # Grid-based training with RAM observations
├── mario_pixel.py              # Pixel-based training (traditional approach)
├── pyproject.toml              # Project dependencies and configuration
├── results/                    # Training results and experiments
│   └── ppo/                   # PPO experiment folders
├── nes-py/                     # Gymnasium port of NES emulator
│   ├── nes_py/                 # Core emulator package
│   ├── README.md               # Emulator documentation
│   └── pyproject.toml          # Package configuration
└── gymnasium-super-mario-bros/ # Gymnasium port of Mario environment
    ├── gymnasium_super_mario_bros/  # Core environment package
    ├── README.md               # Environment documentation
    └── pyproject.toml          # Package configuration
```

## Installation

### Prerequisites
- Python 3.10+
- C++ compiler (for building the NES emulator)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/tooichitake/gymnasium-mario.git
cd gymnasium-mario-master
```

2. Install dependencies:
```bash
pip install .
```

### Alternative: Using Poetry

If you prefer using Poetry for development:
```bash
poetry install
```

## Quick Start

### Grid-Based Training (Recommended)
```bash
python mario_grid_jax.py
```

### Pixel-Based Training
```bash
python mario_pixel.py
```

## Grid-Based Observation Space

The grid-based approach converts the game state into a 13x16 grid where:
- `-1`: Empty space or background
- `0`: Neutral objects (platforms)
- `1`: Enemies
- `2`: Mario (player)

This representation is:
- **Efficient**: 200x smaller than pixel observations
- **Interpretable**: Clear semantic meaning for each value
- **Fast**: Enables rapid experimentation

## Training Features
- **8 parallel environments** for faster data collection
- **Automatic checkpointing** every 500k steps
- **Early stopping** after 10M steps without improvement
- **TensorBoard logging** for monitoring progress
- **Video recording** of best performances

## Results

Training results are saved in `results/ppo/expN/` with:
- `models/`: Saved model checkpoints
- `logs/`: Training logs and TensorBoard files
- `videos/`: Recorded gameplay videos

## Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir results/ppo/expN/logs/tensorboard
```

## Credits

### Original Authors
- **Christian Kauten** - Creator of [nes-py](https://github.com/Kautenja/nes-py) and [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)

### This Implementation
- **tooichitake** - Grid-based approach and Gymnasium ports

### Technologies Used
- [Gymnasium](https://gymnasium.farama.org/) - RL environment API
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [JAX](https://github.com/google/jax) - High-performance computing

## License

- NES-Py: MIT License
- Gymnasium-Super-Mario-Bros: Proprietary - Free for Educational Use
- This implementation: MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use this code in your research, please cite:
```bibtex
@misc{gymnasium-mario,
  author = {tooichitake},
  title = {Mario RL Training with Grid-Based Observations},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/tooichitake/gymnasium-mario}
}
```