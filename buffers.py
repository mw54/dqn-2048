import torch

DEVICE = torch.device("cpu")
torch.set_grad_enabled(False)

class Buffer:
    def __init__(self, buffer_size:int, board_size:int, alpha:float, beta:float, temperature:float):
        self.buffer_size = buffer_size
        self.board_size = board_size
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.position = 0
        self.size = 0
        self.max_priority = 1.0
        
        self.this_states = torch.zeros((buffer_size, board_size, board_size), dtype=torch.int, device=DEVICE)
        self.actions = torch.zeros((buffer_size, 2), dtype=torch.bool, device=DEVICE)
        self.next_states = torch.zeros((buffer_size, board_size, board_size), dtype=torch.int, device=DEVICE)
        self.rewards = torch.zeros(buffer_size, dtype=torch.int, device=DEVICE)
        self.terminals = torch.zeros(buffer_size, dtype=torch.bool, device=DEVICE)
        self.priorities = torch.zeros(buffer_size, device=DEVICE)

    def save(self, path:str) -> dict[str,torch.Tensor]:
        state_dict = {
            "buffer_size": self.buffer_size,
            "board_size": self.board_size,
            "position": self.position,
            "size": self.size,
            "max_priority": self.max_priority,
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
        self.buffer_size = state_dict["buffer_size"]
        self.board_size = state_dict["board_size"]
        self.position = state_dict["position"]
        self.size = state_dict["size"]
        self.max_priority = state_dict["max_priority"]
        
        self.this_states = state_dict["this_states"].to(DEVICE)
        self.actions = state_dict["actions"].to(DEVICE)
        self.next_states = state_dict["next_states"].to(DEVICE)
        self.rewards = state_dict["rewards"].to(DEVICE)
        self.terminals = state_dict["terminals"].to(DEVICE)
        self.priorities = state_dict["priorities"].to(DEVICE)

    def clear(self):
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

        self.this_states.zero_()
        self.actions.zero_()
        self.next_states.zero_()
        self.rewards.zero_()
        self.terminals.zero_()
        self.priorities.zero_()
    
    def push(self, this_states:torch.Tensor, actions:torch.Tensor, next_states:torch.Tensor, rewards:torch.Tensor, terminals:torch.Tensor):
        assert len(this_states) == len(actions) == len(next_states) == len(rewards) == len(terminals)
        size = len(this_states)

        indices = torch.arange(self.position, self.position + size, dtype=torch.int, device=DEVICE) % self.buffer_size
        
        self.this_states[indices] = this_states.to(DEVICE, torch.int)
        self.actions[indices] = actions.to(DEVICE, torch.bool)
        self.next_states[indices] = next_states.to(DEVICE, torch.int)
        self.rewards[indices] = rewards.to(DEVICE, torch.int)
        self.terminals[indices] = terminals.to(DEVICE, torch.bool)
        self.priorities[indices] = self.max_priority

        self.position = (self.position + size) % self.buffer_size
        self.size = min(self.buffer_size, self.size + size)
    
    def sample(self, batch_size:int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.priorities / torch.sum(self.priorities, dtype=torch.float)
        indices = torch.multinomial(p, batch_size, replacement=True)
        weights = (self.buffer_size * p[indices])**(-self.beta)
        weights = weights / torch.sum(weights, dtype=torch.float)
        batch = (self.this_states[indices], self.actions[indices], self.next_states[indices], self.rewards[indices], self.terminals[indices])
        return batch, weights, indices
    
    def update(self, indices:torch.Tensor, errors:torch.Tensor):
        indices = indices.to(DEVICE, torch.long)
        errors = errors.to(DEVICE, torch.float)

        gamma = torch.mean(errors, dtype=torch.float)
        priorities = torch.exp(errors / (self.temperature * gamma))
        self.priorities[indices] = self.priorities[indices]**self.alpha * priorities**(1 - self.alpha)
        self.max_priority = torch.max(self.priorities)

    def __len__(self):
        return self.buffer_size
    
    def __getitem__(self, indices:torch.Tensor) -> tuple[torch.Tensor]:
        indices = indices.to(DEVICE, torch.long)
        return (self.this_states[indices], self.actions[indices], self.next_states[indices], self.rewards[indices], self.terminals[indices])
