import torch

DEVICE = torch.device("cpu")
torch.set_grad_enabled(False)

class Buffer:
    def __init__(self, capacity:int, board_size:int, alpha:float, beta:float, temperature:float, path:str=None):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.head = 0
        self.size = 0
        
        self.this_states = torch.zeros((capacity, board_size, board_size), dtype=torch.int, device=DEVICE)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=DEVICE)
        self.next_states = torch.zeros((capacity, board_size, board_size), dtype=torch.int, device=DEVICE)
        self.rewards = torch.zeros(capacity, dtype=torch.int, device=DEVICE)
        self.terminals = torch.zeros(capacity, dtype=torch.bool, device=DEVICE)
        self.priorities = torch.zeros(capacity, device=DEVICE)

        if path is not None:
            self.load(path)

    def save(self, path:str) -> dict[str,torch.Tensor]:
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
        
        self.this_states[:self.size] = state_dict["this_states"][:self.size].to(DEVICE)
        self.actions[:self.size] = state_dict["actions"][:self.size].to(DEVICE)
        self.next_states[:self.size] = state_dict["next_states"][:self.size].to(DEVICE)
        self.rewards[:self.size] = state_dict["rewards"][:self.size].to(DEVICE)
        self.terminals[:self.size] = state_dict["terminals"][:self.size].to(DEVICE)
        self.priorities[:self.size] = state_dict["priorities"][:self.size].to(DEVICE)

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

        indices = torch.arange(self.head, self.head + size, dtype=torch.int, device=DEVICE) % self.capacity
        
        self.this_states[indices] = this_states.to(DEVICE, torch.int)
        self.actions[indices] = actions.to(DEVICE, torch.long)
        self.next_states[indices] = next_states.to(DEVICE, torch.int)
        self.rewards[indices] = rewards.to(DEVICE, torch.int)
        self.terminals[indices] = terminals.to(DEVICE, torch.bool)
        self.priorities[indices] = 1.0

        self.head = (self.head + size) % self.capacity
        self.size = min(self.capacity, self.size + size)
    
    def sample(self, batch_size:int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.priorities / torch.sum(self.priorities, dtype=torch.float)
        indices = torch.multinomial(p, batch_size, replacement=True)
        weights = (self.capacity * p[indices])**(-self.beta)
        weights = weights / torch.sum(weights, dtype=torch.float)
        batch = (self.this_states[indices], self.actions[indices], self.next_states[indices], self.rewards[indices], self.terminals[indices])
        return batch, weights, indices
    
    def update(self, indices:torch.Tensor, errors:torch.Tensor):
        indices = indices.to(DEVICE, torch.long)
        errors = errors.to(DEVICE, torch.float)

        gamma = torch.mean(errors, dtype=torch.float)
        priorities = torch.exp(-gamma / (self.temperature * errors))
        self.priorities[indices] = self.priorities[indices]**self.alpha * priorities**(1 - self.alpha)

    def __len__(self):
        return self.size
    
    def __getitem__(self, indices:torch.Tensor) -> tuple[torch.Tensor]:
        indices = indices.to(DEVICE, torch.long)
        return (self.this_states[indices], self.actions[indices], self.next_states[indices], self.rewards[indices], self.terminals[indices])
