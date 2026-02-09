import environment
import networks
import agent
import buffer
import multiprocessing as mp
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

class History:
    def __init__(self):
        self.data = dict()

    def register(self, data:dict[str,float]):
        for key, val in data.items():
            if key not in self.data:
                self.data[key] = [val]
            else:
                self.data[key].append(val)
    
    def plot(self, path):
        for key, val in self.data.items():
            plt.figure(figsize=(5, 5), dpi=300)
            plt.plot(val)
            plt.xlabel("step")
            plt.ylabel(key)
            plt.tight_layout()
            plt.savefig(f"{path}/{key}.png")
            plt.close()
    
def collect(data_queue:mp.Queue, model_queue:mp.Queue, environment_params:dict[str,], policy_params:dict[str,]):
    pol = networks.Policy(**policy_params).to(environment.DEVICE)
    env = environment.BatchBoards(**environment_params)
    pol.eval()
    while True:
        if env.terminals.any():
            env.reset()
        if not model_queue.empty():
            pol.load_state_dict(model_queue.get())
        this_states = torch.clone(env.boards)
        actions = pol.act(this_states, stochastic=True)
        rewards = env(actions)
        next_states = torch.clone(env.boards)
        terminals = torch.clone(env.terminals)
        data_queue.put((this_states.cpu(), actions.cpu(), next_states.cpu(), rewards.cpu(), terminals.cpu()))

def optimize(data_queue:mp.Queue, model_queue:mp.Queue, agent_params:dict[str,], buffer_params:dict[str,], total_steps:int, update_interval:int, plot_interval:int, save_interval:int, output_path:str):
    agt = agent.Agent(**agent_params)
    buf = buffer.Buffer(**buffer_params)
    history = History()
    agt.policy.train()
    while buf.size < agt.batch_size:
        data = data_queue.get()
        buf.push(*data)
    for i in tqdm(range(total_steps), desc="step"):
        if not data_queue.empty():
            data = data_queue.get()
            buf.push(*data)
        
        batch, weights, indices = buf.sample(agt.batch_size)
        errors, q1, q2, h = agt.step(*batch, weights=weights)
        buf.update(indices, errors)
        history.register({"error": errors.mean().item(), "q1": q1.mean().item(), "q2": q2.mean().item(), "entropy": h.mean().item()})
        
        if (1 + 1) % update_interval == 0:
            model_queue.put(agt.policy.state_dict())
        if (i + 1) % plot_interval == 0:
            history.plot(output_path)
        if (i + 1) % save_interval == 0:
            buf.save(f"{output_path}/buffer.pt")
            agt.save(f"{output_path}/agent.pt")
