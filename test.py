import torch
import tabulate
import environment
import agent
import constants
import terminal

ACTIONS = {0: "left", 1: "right", 2: "up", 3: "down"}
env = environment.BatchBoards(4, 1)
agt = agent.Agent(**constants.agent_params)
agt.load("agent.pt")
pol = agt.policy.to(env.device)
pol.eval()

with terminal.Terminal() as terminal:
    while not all(env.terminals):
        print(env)
        this_states = torch.clone(env.boards)
        q1, q2 = pol(this_states)
        print(tabulate.tabulate(
            [q1[0].tolist(), q2[0].tolist()],
            headers=["left", "right", "up", "down"],
            showindex=["Q1", "Q2"],
            floatfmt=".2f"
        ))
        print()

        actions = pol.act(this_states)
        print(f"Action: {ACTIONS[actions[0].item()]}")
        env(actions)
        terminal.sleep(0.1)
        terminal.clear()
    
    print(env)
    terminal.sleep(True)
