policy_params = {
    "model_channels": 144,
    "seq_len": 16,
    "num_heads": 6,
    "num_layers": 3,
    "dropout": 0.0,
    "temperature": 1.0
}

agent_params = {
    "policy_params": policy_params,
    "optimizer_params": {
        "lr": 1e-3,
        "weight_decay": 1e-3
    },
    "batch_size": 1024,
    "discount": 0.999,
    "polyak": 0.001
}

environment_params = {
    "board_size": 4,
    "batch_size": 64
}

buffer_params = {
    "capacity": 1048576,
    "board_size": 4,
    "alpha": 0.6,
    "beta": 0.4,
    "temperature": 4.0
}

queue_size = 16
agent_device = "mps"
environment_device = "mps"

collect_params = {
    "environment_params": environment_params,
    "policy_params": policy_params
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
