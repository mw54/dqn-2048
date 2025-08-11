import torch
import tabulate
import environment
import agent
import train

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
        this_states = env.boards.to(agent.DEVICE, copy=True)
        actions, q, p = agt(this_states, pq=True)
        actions = actions.to(environment.DEVICE, copy=True)
        rewards = env(actions).tolist()

        print(tabulate.tabulate(
            [q[0].tolist(), p[0].tolist()],
            headers=["left", "right", "up", "down"],
            showindex=["Q values", "P values"],
            floatfmt=".2f"
        ))

        if pause:
            input()
        print(f"Action: {action_decode(actions)[0]}")
        print(f"Reward: {rewards[0]}")
        print(env)

env = environment.BatchBoards(4, 1)
train.agt.load("agent.pt", True)
train.agt.temperature = 1.0
test(train.agt, env, True)
