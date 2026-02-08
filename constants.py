policy_params = {
    "model_channels": 144,
    "seq_len": 16,
    "num_heads": 3,
    "num_layers": 6,
    "dropout": 0.1,
    "temperature": 64.0
}

agent_params = {
    "policy_params": policy_params,
    "optimizer_params": {
        "lr": 3e-4,
        "weight_decay": 1e-4
    },
    "batch_size": 1024,
    "discount": 0.99,
    "polyak": 0.01
}

environment_params = {
    "board_size": 4,
    "batch_size": 256
}

buffer_params = {
    "capacity": 1048576,
    "board_size": 4,
    "alpha": 0.6,
    "beta": 0.4,
    "temperature": 4.0
}

queue_size = 4096

collect_params = {
    "environment_params": environment_params,
    "policy_params": policy_params
}

optimize_params = {
    "agent_params": agent_params,
    "buffer_params": buffer_params,
    "total_steps": 1048576,
    "update_interval": 64,
    "plot_interval": 1024,
    "save_interval": 64,
    "output_path": "."
}
