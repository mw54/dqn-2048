import environment
import agent
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

def generate_samples(agt:agent.Agent, env:environment.BatchBoards):
    if any(env.terminals):
        env.reset()
    available_actions = env.actions.to(agent.DEVICE, copy=True)
    this_states = env.boards.to(agent.DEVICE, copy=True)
    actions = agt(this_states, available_actions).to(environment.DEVICE, copy=True)
    rewards = env(actions).to(agent.DEVICE, copy=True)
    next_states = env.boards.to(agent.DEVICE, copy=True)
    terminals = env.terminals.to(agent.DEVICE, copy=True)
    agt.update_buffer(this_states, actions.to(agent.DEVICE, copy=True), rewards, next_states, terminals)
