import torch
from tqdm import tqdm
import agent
import buffers
import environment
import utils

def online(agt:agent.Agent, buf:buffers.Buffer, env:environment.BatchBoards, num_epochs:int, epoch_steps:int):
    agt.main.train()
    losses = utils.History()
    maxqs = utils.History()
    for epoch in tqdm(range(num_epochs), desc=f"Epoch"):
        for _ in range(epoch_steps):
            utils.generate_samples(agt, env, buf)
            l, q = utils.step_main(agt, buf)
            losses.stage(l)
            maxqs.stage(q)
        agt.update()
        agt.save("agent.pt")
        buf.save("buffer.pt")
        losses.commit()
        maxqs.commit()
        utils.plot(losses.data, "Loss", "losses.png")
        utils.plot(maxqs.data, "Max Q", "maxqs.png")
        utils.hist(buf.priorities.cpu(), "Priority", "priority.png")
        agt.temperature = 990.0 / (1 + torch.e**(0.0015 * (epoch - 5000))) + 10.0
    return agt

agt = agent.Agent(
    network="ConvNet",
    network_args={
        "input_channel": 1,
        "input_size": 16,
        "hidden_channels": [4, 16, 64],
        "hidden_sizes": [256, 64],
        "output_size": 4,
        "activation": "ReLU"
    },
    optimizer="Adam",
    optimizer_args={
        "lr": 1e-4,
        "amsgrad": True
    },
    batch_size=1000,
    discount=0.99,
    temperature=1000.0
)

buf = buffers.Buffer(
    buffer_size=1000000,
    board_size=4,
    alpha=0.6,
    beta=0.4,
    temperature=4.0
)

if __name__ == "__main__":
    torch.set_num_threads(64)
    env = environment.BatchBoards(4, 100)
    # agt.load("agent.pt")
    # buf.load("buffer.pt")
    agt = online(agt, buf, env, 10000, 1000)
