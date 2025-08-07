import torch
from tqdm import tqdm
import environment
import agent
import utils

def online(agt:agent.Agent, env:environment.BatchBoards, num_epochs:int, epoch_steps:int):
    agt.main.train()
    losses = list()
    maxqs = list()
    for _ in tqdm(range(num_epochs), desc=f"Epoch"):
        loss = 0
        maxq = 0
        for _ in range(epoch_steps):
            utils.generate_samples(agt, env)
            l, q = agt.update_main()
            loss += l
            maxq += q
        losses.append(loss / epoch_steps)
        maxqs.append(maxq / epoch_steps)
        agt.update_target()
        agt.save("agent.pt", False)
        utils.plot(losses, "Loss", "losses.png")
        utils.plot(maxqs, "Max Q", "maxqs.png")
        utils.hist(agt.buffer.priorities, "Priority", "priority.png")
    return agt, losses, maxqs

agt = agent.Agent(
    network="DuelingMLP",
    network_args={
        "input_size": 16,
        "embed_hidden": [1024],
        "embed_size": 1024,
        "value_hidden": [256],
        "advantage_hidden": [256],
        "output_size": 4,
        "activation": "SiLU"
    },
    optimizer="Adam",
    optimizer_args={
        "lr": 1e-3,
        "amsgrad": True
    },
    buffer_args={
        "buffer_size": 1000000,
        "board_size": 4,
        "alpha": 0.6,
        "beta": 0.4,
        "temperature": 10.0
    },
    batch_size=1000,
    discount=0.96,
    temperature=100.0
)

if __name__ == "__main__":
    torch.set_num_threads(16)
    env = environment.BatchBoards(4, 100)
    agt, _, _ = online(agt, env, 10000, 100)
