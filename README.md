# 2048 DQN

Deep Q-Network with transformer architecture to solve the 2048 game.

## Architecture

- **Agent**: Double Q-learning with entropy regularization
- **Network**: Transformer encoder (3 layers, 4 heads)
- **Replay**: Prioritized experience replay (1M capacity)
- **Training**: Parallel data collection with asynchronous optimization

## Usage

### Training

```bash
python main.py
```

Training runs with:
- 256 parallel environments for data collection
- 1024 batch size for optimization
- Model checkpoints saved every 1024 steps

### Testing

```bash
python test.py
```

Plays a single game using the trained agent with Q-value visualization.

### Interactive Play

```bash
python 2048.py
```

Play 2048 manually in the terminal.

## Structure

```
2048.py         # Single-board game environment
environment.py  # Vectorized batch environment
networks.py     # Transformer-based Q-networks
agent.py        # DQN agent with target network
buffer.py       # Prioritized experience replay
processes.py    # Data collection and optimization loops
main.py         # Training entry point
test.py         # Agent evaluation
constants.py    # Hyperparameters
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Discount (γ) | 0.99 |
| Polyak (τ) | 0.01 |
| Batch size | 1024 |
| Buffer size | 1M |
| Temperature | 64.0 |
| Model channels | 96 |

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- matplotlib
- tabulate
- tqdm

## Device

Currently configured for MPS (Apple Silicon). Modify `DEVICE` in `agent.py` and `environment.py` for CUDA or CPU.
