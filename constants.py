policy_params = {
    "model_channels": 144,
    "seq_len": 16,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.0
}

agent_params = {
    "policy_params": policy_params,
    "optimizer_params": {
        "lr": 1e-3,
        "weight_decay": 1e-6
    },
    "batch_size": 256,
    "discount": 0.99,
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
    "update_interval": 256,
    "plot_interval": 4096,
    "save_interval": 65536,
    "output_path": "."
}
