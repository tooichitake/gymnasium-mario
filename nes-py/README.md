# nes-py

A Nintendo Entertainment System (NES) emulator and [Gymnasium](https://gymnasium.farama.org/) interface for Python. This is a gymnasium-compatible port of the original [nes-py](https://github.com/Kautenja/nes-py) library, updated to work with the modern Gymnasium reinforcement learning framework.

## Description

nes-py provides a Python interface to interact with NES games through the Gymnasium API. It includes a fully-featured NES emulator written in C++ with Python bindings, allowing you to:

- Load and play NES ROM files
- Create Gymnasium environments for reinforcement learning
- Access game states, render frames, and control inputs programmatically
- Build custom environments for specific NES games

## Credits

This library is based on the original work by **Christian Kauten**. The original nes-py was designed for OpenAI Gym, and this version has been updated to support the newer Gymnasium framework while maintaining backward compatibility where possible.

## Installation

### From Source

```bash
# Clone the repository
git clone <repository-url>
cd nes-py

# Install in development mode
pip install -e .
```

### Dependencies

The package will automatically install the following dependencies:
- gymnasium >= 1.1.1
- numpy >= 2.2.0
- pyglet >= 2.1.3
- tqdm >= 4.48.2

## Quick Usage Example

```python
import gymnasium as gym
from nes_py import NESEnv

# Create an NES environment with your ROM file
env = NESEnv('path/to/your/game.nes')

# Reset the environment
observation, info = env.reset()

# Run a simple game loop
done = False
while not done:
    # Take a random action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Render the game (optional)
    env.render()

env.close()
```

### Using with Gymnasium Wrappers

```python
import gymnasium as gym
from nes_py import NESEnv
from nes_py.wrappers import JoypadSpace

# Create environment with simplified action space
env = NESEnv('path/to/your/game.nes')
env = JoypadSpace(env, [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
])

# Use standard Gymnasium wrappers
env = gym.wrappers.FrameStack(env, 4)
env = gym.wrappers.GrayScaleObservation(env)
```

## Differences from Original Gym Version

1. **API Updates**: 
   - Uses `env.reset()` returning `(observation, info)` instead of just `observation`
   - Uses `env.step()` returning `(obs, reward, terminated, truncated, info)` instead of `(obs, reward, done, info)`
   - Compatible with Gymnasium >= 1.0.0

2. **Import Changes**:
   - Import from `gymnasium` instead of `gym`
   - All wrappers and spaces use Gymnasium's implementation

3. **Render API**:
   - Updated to support Gymnasium's rendering system
   - Supports both human and rgb_array render modes

## Features

### NES Emulator Features
- **Accurate NES emulation**: Full support for NES CPU, PPU, and APU
- **Mapper support**: Supports common NES cartridge mappers (NROM, CNROM, SxROM, UxROM)
- **Save states**: Save and load game states for reproducible experiments
- **Frame-perfect accuracy**: Deterministic emulation suitable for RL research

### Python Interface Features
- **Gymnasium compatibility**: Full integration with modern RL frameworks
- **Customizable action spaces**: Use JoypadSpace wrapper to define custom button combinations
- **Rendering options**: Human-viewable window or RGB array for processing
- **Performance**: Efficient C++ core with Python bindings

## Requirements

### System Requirements
- Python 3.10 or higher
- C++ compiler (for building from source)
- CMake (for building the C++ extension)

### Python Dependencies
- gymnasium >= 1.1.1
- numpy >= 2.2.0
- pyglet >= 2.1.3 (for rendering)
- tqdm >= 4.48.2

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2019 Christian Kauten

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## Acknowledgments

- **Christian Kauten** - Original author of nes-py
- **OpenAI Gym/Farama Gymnasium** - For providing the standard RL environment interface
- **NES Development Community** - For extensive documentation on NES hardware
- All contributors who helped port this library to Gymnasium

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### Common Issues

1. **ImportError: Missing static lib_nes_env*.so library**
   - Ensure you have built the C++ extension properly
   - Try reinstalling with `pip install -e .`

2. **Rendering issues on headless systems**
   - Set environment variable: `export DISPLAY=:0`
   - Or use `xvfb-run` for virtual display

3. **ROM compatibility issues**
   - Ensure your ROM file is valid
   - Check if the mapper is supported (NROM, CNROM, SxROM, UxROM)