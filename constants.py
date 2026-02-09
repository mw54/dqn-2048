policy_params = {
    "model_channels": 96,
    "seq_len": 16,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.0,
    "temperature": 16.0
}

agent_params = {
    "policy_params": policy_params,
    "optimizer_params": {
        "lr": 1e-3,
        "weight_decay": 1e-6
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

queue_size = 64

collect_params = {
    "environment_params": environment_params,
    "policy_params": policy_params
}

optimize_params = {
    "agent_params": agent_params,
    "buffer_params": buffer_params,
    "total_steps": 1048576,
    "update_interval": 64,
    "plot_interval": 256,
    "save_interval": 1024,
    "output_path": "."
}
