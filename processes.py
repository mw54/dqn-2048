import environment
import networks
import agent
import buffer
import multiprocessing as mp
import torch
from tqdm import tqdm
    
def collect(data_queue:mp.Queue, model_queue:mp.Queue, environment_params:dict[str,], policy_params:dict[str,], temperature:float):
    env = environment.BatchBoards(**environment_params)
    pol = networks.Policy(**policy_params).to(env.device)
    pol.eval()
    while True:
        if env.terminals.any():
            env.reset()
        if not model_queue.empty():
            state_dict = model_queue.get()
            pol.load_state_dict(state_dict)
        this_states = torch.clone(env.boards)
        actions = pol.act(this_states, temperature=temperature)
        rewards = env(actions)
        next_states = torch.clone(env.boards)
        terminals = torch.clone(env.terminals)
        data_queue.put((this_states.cpu(), actions.cpu(), next_states.cpu(), rewards.cpu(), terminals.cpu()))

def optimize(data_queue:mp.Queue, model_queue:mp.Queue, agent_params:dict[str,], buffer_params:dict[str,], total_steps:int, update_interval:int, plot_interval:int, save_interval:int, output_path:str):
    agt = agent.Agent(**agent_params)
    buf = buffer.Buffer(**buffer_params)
    agt.policy.train()
    while buf.size < agt.batch_size:
        data = data_queue.get()
        buf.push(*data)
    for i in tqdm(range(total_steps), desc="step"):
        while not data_queue.empty():
            data = data_queue.get()
            buf.push(*data)
        
        batch, weights, indices = buf.sample(agt.batch_size)
        errors = agt.step(*batch, weights=weights)
        buf.update(indices, errors)
        
        if (i + 1) % update_interval == 0:
            state_dict = {k: v.cpu() for k, v in agt.policy.state_dict().items()}
            model_queue.put(state_dict)
        if (i + 1) % plot_interval == 0:
            agt.plot(output_path)
        if (i + 1) % save_interval == 0:
            buf.save(f"{output_path}/buffer.pt")
            agt.save(f"{output_path}/agent.pt")
