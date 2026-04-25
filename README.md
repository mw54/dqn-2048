# DQN 2048

Deep Q-Network with transformer architecture to solve the 2048 game.

## Architecture

- **Agent**: Double Q-learning
- **Network**: Transformer Encoder (3 layers, 4 heads) + MLP Value Head (2 layers, 256 hidden channels)
- **Replay**: Prioritized experience replay (1M capacity)
- **Training**: Parallel data collection with asynchronous optimization

## Usage

### Training

```bash
python main.py
```

Training runs with:
- 256 parallel environments for data collection
- 256 batch size for optimization
- Model checkpoints saved every 16384 steps

### Testing

```bash
python test.py
```

Plays a single game using the trained agent with Q-value visualization.

### Interactive Play

```bash
python environment.py
```

Play the game manually in the terminal.

## Structure

```
environment.py  # Vectorized batch environment
networks.py     # Transformer-based Q-networks
agent.py        # DQN agent with target network
buffer.py       # Prioritized experience replay
processes.py    # Data collection and optimization loops
main.py         # Training entry point
test.py         # Agent evaluation
terminal.py     # Terminal utilities
constants.py    # Hyperparameters
```

## Hyperparameters

| Parameter      | Value |
|----------------|-------|
| Learning rate  | 1e-3  |
| Discount (γ)   | 0.996 |
| Polyak (τ)     | 0.004 |
| Batch size     | 256   |
| Buffer size    | 1M    |
| Temperature    | 4.0   |
| Model channels | 96    |

## Requirements

- Python 3.13+
- PyTorch 2.7+
- NumPy
- matplotlib
- tabulate
- tqdm

## Device

Currently configured for MPS (Apple Silicon). Modify `agent_device` and `environment_device` in `constants.py` for CUDA or CPU.
