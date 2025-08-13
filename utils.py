import torch
import matplotlib.pyplot as plt
import agent
import buffers
import environment

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

def generate_samples(agt:agent.Agent, env:environment.BatchBoards, buf:buffers.Buffer):
    if any(env.terminals):
        env.reset()
    this_states = torch.clone(env.boards)
    actions = agt(this_states)
    rewards = env(actions)
    next_states = torch.clone(env.boards)
    terminals = torch.clone(env.terminals)
    buf.push(this_states, actions, next_states, rewards, terminals)

def step_main(agt:agent.Agent, buf:buffers.Buffer) -> tuple[float, float]:
    batch, weights, indices = buf.sample(agt.batch_size)
    errors, q = agt.step(*batch, weights)
    buf.update(indices, errors)
    return torch.mean(errors).item(), torch.max(q).item()

class History:
    def __init__(self):
        self.data = list()
        self.temp = list()

    def stage(self, value:float):
        self.temp.append(value)
    
    def commit(self):
        self.data.append(sum(self.temp) / len(self.temp))
        self.temp.clear()
