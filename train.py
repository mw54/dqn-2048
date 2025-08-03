import environment
import agent
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot(losses:list[float], ylabel:str, path:str):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.savefig(path)
    plt.close()

def hist(data:list[float], xlabel:str, path:str):
    plt.figure(figsize=(10, 6), dpi=300)
    plt.hist(data, bins=100, edgecolor="white", log=True)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.savefig(path)
    plt.close()

def online(agt:agent.Agent, env:environment.BatchBoards, num_epochs:int, epoch_steps:int):
    agt.main.train()
    losses = list()
    maxqs = list()
    for _ in tqdm(range(num_epochs), desc=f"Epoch"):
        loss = 0
        maxq = 0
        for _ in range(epoch_steps):
            if any(env.terminals):
                env.reset()
            available_actions = env.actions.to(agent.DEVICE, copy=True)
            this_states = env.boards.to(agent.DEVICE, copy=True)
            actions = agt(this_states, available_actions).to(environment.DEVICE, copy=True)
            rewards = env(actions).to(agent.DEVICE, copy=True)
            next_states = env.boards.to(agent.DEVICE, copy=True)
            terminals = env.terminals.to(agent.DEVICE, copy=True)
            agt.update_buffer(this_states, actions.to(agent.DEVICE, copy=True), rewards, next_states, terminals)
            l, q = agt.update_main()
            loss += l
            maxq += q
        losses.append(loss / epoch_steps)
        maxqs.append(maxq / epoch_steps)
        agt.update_target()
        agt.save("agent.pt", False)
        plot(losses, "Loss", "losses.png")
        plot(maxqs, "Max Q", "maxqs.png")
        hist(agt.buffer.priorities, "Priority", "priority.png")
    return agt, losses, maxqs

env = environment.BatchBoards(4, 64)
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
        "buffer_size": 1048576,
        "board_size": 4,
        "alpha": 0.6,
        "beta": 0.4,
        "temperature": 10.0
    },
    batch_size=1024,
    discount=0.999,
    temperature=100.0
)

agt, _, _ = online(agt, env, 4096, 256)
