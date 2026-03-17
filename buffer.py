import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

class Buffer:
    def __init__(self, capacity:int, board_size:int, alpha:float, beta:float, temperature:float, device:str="cpu", path:str=None):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.device = device
        self.head = 0
        self.size = 0
        
        self.this_states = torch.zeros((capacity, board_size, board_size), dtype=torch.int, device=self.device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.next_states = torch.zeros((capacity, board_size, board_size), dtype=torch.int, device=self.device)
        self.rewards = torch.zeros(capacity, dtype=torch.int, device=self.device)
        self.terminals = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.priorities = torch.zeros(capacity, device=self.device)

        if path is not None:
            self.load(path)

    def save(self, path:str):
        state_dict = {
            "head": self.head,
            "size": self.size,
            "this_states": self.this_states,
            "actions": self.actions,
            "next_states": self.next_states,
            "rewards": self.rewards,
            "terminals": self.terminals,
            "priorities": self.priorities
        }
        torch.save(state_dict, path)
    
    def load(self, path:str):
        state_dict = torch.load(path, weights_only=False, map_location="cpu")
        self.size = min(self.capacity, state_dict['size'])
        self.head = min(self.capacity, state_dict['head']) % self.capacity
        
        self.this_states[:self.size] = state_dict["this_states"][:self.size].to(self.device)
        self.actions[:self.size] = state_dict["actions"][:self.size].to(self.device)
        self.next_states[:self.size] = state_dict["next_states"][:self.size].to(self.device)
        self.rewards[:self.size] = state_dict["rewards"][:self.size].to(self.device)
        self.terminals[:self.size] = state_dict["terminals"][:self.size].to(self.device)
        self.priorities[:self.size] = state_dict["priorities"][:self.size].to(self.device)

    def clear(self):
        self.head = 0
        self.size = 0

        self.this_states.zero_()
        self.actions.zero_()
        self.next_states.zero_()
        self.rewards.zero_()
        self.terminals.zero_()
        self.priorities.zero_()
    
    def push(self, this_states:torch.Tensor, actions:torch.Tensor, next_states:torch.Tensor, rewards:torch.Tensor, terminals:torch.Tensor):
        assert len(this_states) == len(actions) == len(next_states) == len(rewards) == len(terminals)
        size = len(this_states)

        indices = torch.arange(self.head, self.head + size, dtype=torch.int, device=self.device) % self.capacity
        
        self.this_states[indices] = this_states.to(self.device, torch.int)
        self.actions[indices] = actions.to(self.device, torch.long)
        self.next_states[indices] = next_states.to(self.device, torch.int)
        self.rewards[indices] = rewards.to(self.device, torch.int)
        self.terminals[indices] = terminals.to(self.device, torch.bool)
        self.priorities[indices] = 1.0

        self.head = (self.head + size) % self.capacity
        self.size = min(self.capacity, self.size + size)
    
    def sample(self, batch_size:int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = F.normalize(self.priorities, dim=0, p=1)
        indices = torch.multinomial(p, batch_size, replacement=True)
        weights = (self.capacity * p[indices])**(-self.beta)
        weights = weights / torch.sum(weights, dtype=torch.float)
        batch = (self.this_states[indices], self.actions[indices], self.next_states[indices], self.rewards[indices], self.terminals[indices])
        return batch, weights, indices
    
    def update(self, indices:torch.Tensor, errors:torch.Tensor):
        indices = indices.to(self.device, torch.long)
        errors = errors.to(self.device, torch.float)

        gamma = torch.mean(errors, dtype=torch.float)
        priorities = torch.exp(-gamma / (self.temperature * errors))
        self.priorities[indices] = self.priorities[indices]**self.alpha * priorities**(1 - self.alpha)

    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, indices:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = indices.to(self.device, torch.long)
        return (self.this_states[indices], self.actions[indices], self.next_states[indices], self.rewards[indices], self.terminals[indices])
