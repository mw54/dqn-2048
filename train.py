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
        agt.temperature = max(agt.temperature * 0.996, 10.0)
        utils.plot(losses, "Loss", "losses.png")
        utils.plot(maxqs, "Max Q", "maxqs.png")
        utils.hist(agt.buffer.priorities, "Priority", "priority.png")
    return agt, losses, maxqs

agt = agent.Agent(
    network="MLP",
    network_args={
        "input_size": 16,
        "hidden_sizes": [64],
        "output_size": 4,
        "activation": "PReLU"
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
    discount=0.999,
    temperature=1000.0
)

if __name__ == "__main__":
    torch.set_num_threads(16)
    env = environment.BatchBoards(4, 100)
    agt.load("checkpoints/1/agent.pt", False)
    agt, _, _ = online(agt, env, 10000, 1000)
