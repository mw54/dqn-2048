import torch
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
        available_actions = env.actions.to(agent.DEVICE, copy=True)
        this_states = env.boards.to(agent.DEVICE, copy=True)
        actions, q, p = agt(this_states, available_actions, pq=True)
        actions = actions.to(environment.DEVICE, copy=True)
        rewards = env(actions).tolist()

        print(f"Q values: {[round(val, 2) for val in q.tolist()[0]]}")
        print(f"P values: {[round(val, 4) for val in p.tolist()[0]]}")
        if pause:
            input()
        print(f"Action: {action_decode(actions)}")
        print(f"Reward: {rewards}")
        print(env)

env = environment.BatchBoards(4, 1)
train.agt.load("agent.pt", True)
train.agt.temperature = 1.0
test(train.agt, env, True)
