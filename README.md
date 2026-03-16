# 2048 DQN

Deep Q-Network with transformer architecture to solve the 2048 game.

## Architecture

- **Agent**: Double Q-learning
- **Network**: Transformer Encoder (4 layers, 6 heads)
- **Replay**: Prioritized experience replay (1M capacity)
- **Training**: Parallel data collection with asynchronous optimization

## Usage

### Training

```bash
python main.py
```

Training runs with:
- 64 parallel environments for data collection
- 1024 batch size for optimization
- Model checkpoints saved every 65536 steps

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

| Parameter      | Value |
|----------------|-------|
| Learning rate  | 1e-3  |
| Discount (γ)   | 0.999 |
| Polyak (τ)     | 0.004 |
| Batch size     | 1024  |
| Buffer size    | 1M    |
| Temperature    | 16.0  |
| Model channels | 144   |

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- matplotlib
- tabulate
- tqdm

## Device

Currently configured for MPS (Apple Silicon). Modify `agent_device` and `environment_device` in `constants.py` for CUDA or CPU.
