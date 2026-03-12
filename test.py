import torch
import tabulate
import environment
import agent
import constants

def action_decode(actions:torch.Tensor) -> list[str]:
    mapping = {0: "left", 1: "right", 2: "up", 3: "down"}
    decoded = list()
    for action in actions.tolist():
        decoded.append(mapping[action])
    return decoded

env = environment.BatchBoards(4, 1)
agt = agent.Agent(**constants.agent_params)
agt.load("agent.pt")
pol = agt.policy.to(env.device)
pol.eval()

print(env)
while not all(env.terminals):
    this_states = torch.clone(env.boards)
    q1, q2 = pol(this_states)

    print(tabulate.tabulate(
        [q1[0].tolist(), q2[0].tolist()],
        headers=["left", "right", "up", "down"],
        showindex=["Q1", "Q2"],
        floatfmt=".2f"
    ))

    actions = pol.act(this_states, stochastic=False)
    rewards = env(actions).tolist()
    input()
    print(f"Action: {action_decode(actions)[0]}")
    print(f"Reward: {rewards[0]}")
    print(env)
