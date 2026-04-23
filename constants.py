policy_params = {
    "model_channels": 144,
    "hidden_channels": 1024,
    "num_cells": 16,
    "num_heads": 6,
    "num_layers": 4,
    "dropout": 0.1
}

agent_params = {
    "policy_params": policy_params,
    "optimizer_params": {
        "lr": 1e-3,
        "weight_decay": 1e-4
    },
    "batch_size": 256,
    "discount": 0.996,
    "polyak": 0.004,
    "device": "mps"
}

environment_params = {
    "board_size": 4,
    "batch_size": 64,
    "device": "mps"
}

buffer_params = {
    "capacity": 1048576,
    "board_size": 4,
    "alpha": 0.6,
    "beta": 0.4,
    "temperature": 4.0,
    "device": "cpu"
}

queue_size = 4

collect_params = {
    "environment_params": environment_params,
    "policy_params": policy_params,
    "temperature": 4.0
}

optimize_params = {
    "agent_params": agent_params,
    "buffer_params": buffer_params,
    "total_steps": 1048576,
    "update_interval": 64,
    "plot_interval": 1024,
    "save_interval": 16384,
    "output_path": "."
}
