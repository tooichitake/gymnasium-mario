# gymnasium-super-mario-bros

Super Mario Bros environment for OpenAI Gymnasium - A port of gym-super-mario-bros to the modern Gymnasium API.

## Description

This is a Gymnasium port of the popular gym-super-mario-bros environment, providing a faithful recreation of the classic Nintendo Entertainment System (NES) game for reinforcement learning research. The environment has been updated to work with the latest Gymnasium API while maintaining backward compatibility with existing algorithms.

## Credits

Original gym-super-mario-bros created by **Christian Kauten** ([@Kautenja](https://github.com/Kautenja))
- Original repository: https://github.com/Kautenja/gym-super-mario-bros

Gymnasium port maintained by **Tooichitake** ([@tooichitake](https://github.com/tooichitake))

## Installation

### From GitHub Repository

```bash
git clone https://github.com/tooichitake/gymnasium-mario.git
cd gymnasium-mario/gymnasium-super-mario-bros
pip install -e .
```

### Dependencies Installation

```bash
pip install -r requirements.txt
```

## Quick Start Usage Example

```python
import gymnasium_super_mario_bros
from gymnasium_super_mario_bros.actions import SIMPLE_MOVEMENT

# Create the environment
env = gymnasium_super_mario_bros.make('SuperMarioBros-v0', render_mode='human')

# Optional: Use wrapper for simplified action space
env = gymnasium.wrappers.NormalizeObservation(env)

# Reset the environment
state, info = env.reset()

# Run a simple game loop
done = False
while not done:
    # Take a random action
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
env.close()
```

## Available Environments

### Standard Super Mario Bros Environments
- `SuperMarioBros-v0`: Original ROM with full resolution
- `SuperMarioBros-v1`: Downsampled ROM for faster processing
- `SuperMarioBros-v2`: Pixel-perfect ROM
- `SuperMarioBros-v3`: Rectangle-based ROM

### Random Stage Environments
- `SuperMarioBrosRandomStages-v0`: Random stages with vanilla ROM
- `SuperMarioBrosRandomStages-v1`: Random stages with downsampled ROM
- `SuperMarioBrosRandomStages-v2`: Random stages with pixel ROM
- `SuperMarioBrosRandomStages-v3`: Random stages with rectangle ROM

### Super Mario Bros 2 (Lost Levels)
- `SuperMarioBros2-v0`: Lost Levels with vanilla ROM
- `SuperMarioBros2-v1`: Lost Levels with downsampled ROM

### Individual Stage Environments
You can also create environments for specific stages:
```python
# Format: SuperMarioBros-<world>-<stage>-v<version>
env = gymnasium_super_mario_bros.make('SuperMarioBros-1-1-v0')  # World 1, Stage 1
env = gymnasium_super_mario_bros.make('SuperMarioBros-8-4-v0')  # World 8, Stage 4
```

## Differences from Original gym Version

### API Changes
1. **Gymnasium Compatibility**: Updated to use `gymnasium` instead of deprecated `gym`
2. **Reset Method**: Now returns `(observation, info)` tuple instead of just observation
3. **Step Method**: Returns 5 values: `(obs, reward, terminated, truncated, info)` instead of 4
4. **Render Mode**: Specified during environment creation with `render_mode` parameter

### Example Migration
```python
# Old gym version
import gym
import gym_super_mario_bros
env = gym_super_mario_bros.make('SuperMarioBros-v0')
obs = env.reset()
obs, reward, done, info = env.step(action)

# New gymnasium version
import gymnasium
import gymnasium_super_mario_bros
env = gymnasium_super_mario_bros.make('SuperMarioBros-v0', render_mode='human')
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

## Requirements

- Python >= 3.10
- gymnasium >= 1.1.1
- nes-py >= 8.3.0
- numpy >= 2.2.0
- opencv-python >= 3.4.0.12
- matplotlib >= 2.0.2
- pygame >= 1.9.3
- pyglet >= 2.1.3

## Action Spaces

The environment provides different action space configurations:
- **SIMPLE_MOVEMENT**: Basic movements (left, right, jump)
- **COMPLEX_MOVEMENT**: All possible button combinations
- **RIGHT_ONLY**: Simplified action space for right-only movement

```python
from gymnasium_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
```

## License

Copyright (c) 2018 Christian Kauten

This project is provided for educational purposes only. It is not affiliated with nor approved by Nintendo Co., Ltd.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

## Acknowledgments

- Original Super Mario Bros game by Nintendo Co., Ltd.
- NES emulation powered by [nes-py](https://github.com/Kautenja/nes-py)
- Original gym-super-mario-bros by Christian Kauten
- Gymnasium API by the Farama Foundation
- All contributors who have helped improve this port

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue on the [GitHub repository](https://github.com/tooichitake/gymnasium-mario).

## Citation

If you use this environment in your research, please cite:

```bibtex
@misc{gymnasium-super-mario-bros,
  author = {Christian Kauten and Tooichitake},
  title = {Gymnasium Super Mario Bros},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/tooichitake/gymnasium-mario}},
}
```