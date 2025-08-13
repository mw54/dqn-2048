import torch
import torch.optim as optim
import networks

DEVICE = torch.device("mps")
torch.set_grad_enabled(False)

def transform_states(states:torch.Tensor) -> torch.Tensor:
    states[states == 0] = 1
    states = torch.log2(states)
    return states

def transform_rewards(rewards:torch.Tensor, terminals:torch.Tensor) -> torch.Tensor:
    return rewards

def transform_actions(actions:torch.Tensor) -> torch.Tensor:
    actions = actions[:,0] * 2 + actions[:,1]
    return actions

class Agent:
    def __init__(self, network:str, network_args:dict, optimizer:str, optimizer_args:dict, batch_size:int, discount:float, temperature:float):
        Network = getattr(networks, network)
        Optimizer = getattr(optim, optimizer)
        self.main = Network(**network_args).to(DEVICE)
        self.target = Network(**network_args).to(DEVICE).requires_grad_(False)
        self.optimizer = Optimizer(self.main.parameters(), **optimizer_args)
        self.update()

        self.batch_size = batch_size
        self.discount = discount
        self.temperature = temperature

    def save(self, path:str):
        state_dict = {
            "main": self.main.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(state_dict, path)

    def load(self, path:str):
        state_dict = torch.load(path, weights_only=False, map_location="cpu")
        self.main.load_state_dict(state_dict["main"])
        self.target.load_state_dict(state_dict["target"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def update(self):
        self.target.load_state_dict(self.main.state_dict())
        self.target.eval()

    def step(self, this_states:torch.Tensor, actions:torch.Tensor, next_states:torch.Tensor, rewards:torch.Tensor, terminals:torch.Tensor, weights:torch.Tensor) -> torch.Tensor:
        assert len(this_states) == len(actions) == len(next_states) == len(rewards) == len(terminals) == len(weights)

        this_states = transform_states(this_states).to(DEVICE, torch.float)
        actions = transform_actions(actions).to(DEVICE, torch.long)
        next_states = transform_states(next_states).to(DEVICE, torch.float)
        rewards = transform_rewards(rewards, terminals).to(DEVICE, torch.float)
        terminals = terminals.to(DEVICE, torch.bool)
        weights = weights.to(DEVICE, torch.float)
        
        y = rewards + self.discount * torch.max(self.target(next_states), dim=1)[0] * (~terminals)
        
        with torch.enable_grad():
            self.optimizer.zero_grad()
            q = torch.gather(self.main(this_states), dim=1, index=actions.unsqueeze(1)).squeeze(1)
            errors = (y - q)**2
            loss = torch.dot(weights, errors)
            loss.backward()
            self.optimizer.step()

        return errors, q

    def __call__(self, this_states:torch.Tensor, pq:bool=False) -> torch.Tensor:
        this_states = transform_states(this_states).to(DEVICE, torch.float)
        q = self.main(this_states)
        p = torch.softmax(q / self.temperature, dim=1)
        indices = torch.multinomial(p, num_samples=1).squeeze(1)
        actions = torch.stack([indices // 2, indices % 2], dim=1).to(torch.bool)
        if pq:
            return actions, q, p
        else:
            return actions
