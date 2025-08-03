import torch
import environment
import agent

def action_decode(actions:torch.Tensor) -> list[str]:
    assert actions.size(1) == 2
    mapping = {0: "left", 1: "right", 2: "up", 3: "down"}
    indices = (actions[:,0] * 2 + actions[:,1]).tolist()
    decoded = list()
    for index in indices:
        decoded.append(mapping[index])
    return decoded

def test(agt:agent.Agent, env:environment.BatchBoards, pause:bool):
    agt.main.eval()
    print(env)
    while not all(env.terminals):
        if pause:
            input()
        available_actions = env.actions.to(agent.DEVICE, copy=True)
        this_states = env.boards.to(agent.DEVICE, copy=True)
        actions, q, p = agt(this_states, available_actions, pq=True)
        actions = actions.to(environment.DEVICE, copy=True)
        rewards = env(actions).tolist()

        print(f"Q values: {[round(val, 2) for val in q.tolist()[0]]}")
        print(f"P values: {[round(val, 4) for val in p.tolist()[0]]}")
        print(f"Action: {action_decode(actions)}")
        print(f"Reward: {rewards}")
        print(env)

env = environment.BatchBoards(4, 1)
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
    temperature=1.0
)

agt.load("agent.pt", True)
test(agt, env, True)
