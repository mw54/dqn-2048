import torch
import torch.optim as optim
import networks

DEVICE = torch.device("cpu")
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

class Buffer:
    def __init__(self, buffer_size:int, board_size:int, alpha:float, beta:float, temperature):
        self.buffer_size = buffer_size
        self.board_size = board_size
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.position = 0
        self.size = 0
        
        self.this_states = torch.zeros((buffer_size, board_size, board_size), device=DEVICE)
        self.actions = torch.zeros((buffer_size, 2), dtype=torch.bool, device=DEVICE)
        self.next_states = torch.zeros((buffer_size, board_size, board_size), device=DEVICE)
        self.rewards = torch.zeros(buffer_size, device=DEVICE)
        self.terminals = torch.zeros(buffer_size, dtype=torch.bool, device=DEVICE)
        self.priorities = torch.zeros(buffer_size, device=DEVICE)

    def clear(self):
        self.position = 0
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

        indices = torch.arange(self.position, self.position + size, dtype=torch.int, device=DEVICE) % self.buffer_size
        
        self.this_states[indices] = this_states.to(torch.float)
        self.actions[indices] = actions.to(torch.bool)
        self.next_states[indices] = next_states.to(torch.float)
        self.rewards[indices] = rewards.to(torch.float)
        self.terminals[indices] = terminals.to(torch.bool)
        self.priorities[indices] = 1.0

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
        gamma = torch.mean(errors, dtype=torch.float)
        priorities = torch.exp(-gamma / (self.temperature * errors))
        self.priorities[indices] = self.priorities[indices]**self.alpha * priorities**(1 - self.alpha)

    def __len__(self):
        return self.buffer_size
    
    def __getitem__(self, indices:list[int]) -> tuple[torch.Tensor]:
        return (self.this_states[indices], self.actions[indices], self.next_states[indices], self.rewards[indices], self.terminals[indices])
    
    def state_dict(self) -> dict[str,torch.Tensor]:
        return {
            "buffer_size": self.buffer_size,
            "board_size": self.board_size,
            "position": self.position,
            "size": self.size,
            "this_states": self.this_states,
            "actions": self.actions,
            "next_states": self.next_states,
            "rewards": self.rewards,
            "terminals": self.terminals,
            "priorities": self.priorities
        }
    
    def load_state_dict(self, state_dict:dict[str,torch.Tensor]):
        self.buffer_size = state_dict["buffer_size"]
        self.board_size = state_dict["board_size"]
        self.position = state_dict["position"]
        self.size = state_dict["size"]
        
        self.this_states = state_dict["this_states"].to(DEVICE)
        self.actions = state_dict["actions"].to(DEVICE)
        self.next_states = state_dict["next_states"].to(DEVICE)
        self.rewards = state_dict["rewards"].to(DEVICE)
        self.terminals = state_dict["terminals"].to(DEVICE)
        self.priorities = state_dict["priorities"].to(DEVICE)

    def to(self, device:torch.device):
        self.this_states = self.this_states.to(device)
        self.actions = self.actions.to(device)
        self.next_states = self.next_states.to(device)
        self.rewards = self.rewards.to(device)
        self.terminals = self.terminals.to(device)
        self.priorities = self.priorities.to(device)
        return self

class Agent:
    def __init__(self, network:str, network_args:dict, optimizer:str, optimizer_args:dict, buffer_args:dict, batch_size:int, discount:float, temperature:float):
        Network = getattr(networks, network)
        Optimizer = getattr(optim, optimizer)
        self.main = Network(**network_args).to(DEVICE)
        self.target = Network(**network_args).to(DEVICE).requires_grad_(False)
        self.optimizer = Optimizer(self.main.parameters(), **optimizer_args)
        self.buffer = Buffer(**buffer_args).to(DEVICE)
        self.update_target()

        self.batch_size = batch_size
        self.discount = discount
        self.temperature = temperature

    def save(self, path:str, strip):
        if not strip:
            state_dict = {
                "main": self.main.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "buffer": self.buffer.state_dict()
            }
        else:
            state_dict = {
                "main": self.main.state_dict()
            }
        torch.save(state_dict, path)

    def load(self, path:str, strip):
        state_dict = torch.load(path, weights_only=False, map_location="cpu")

        if strip:
            self.main.load_state_dict(state_dict["main"])
        else:
            self.main.load_state_dict(state_dict["main"])
            self.target.load_state_dict(state_dict["target"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.buffer.load_state_dict(state_dict["buffer"])

    def update_target(self):
        self.target.load_state_dict(self.main.state_dict())
        self.target.eval()

    def update_main(self) -> tuple[float, float]:
        batch, weights, indices = self.buffer.sample(self.batch_size)
        this_states, actions, next_states, rewards, terminals = batch
        
        y = rewards + self.discount * torch.max(self.target(next_states), dim=1)[0] * (~terminals)
        
        with torch.enable_grad():
            self.optimizer.zero_grad()
            q = torch.gather(self.main(this_states), dim=1, index=transform_actions(actions).unsqueeze(1)).squeeze(1)
            errors = (y - q)**2
            loss = torch.dot(weights, errors)
            loss.backward()
            self.optimizer.step()

        self.buffer.update(indices, errors.detach())
        maxq = torch.max(q)
        return loss.item(), maxq.item()

    def update_buffer(self, this_states:torch.Tensor, actions:torch.Tensor, rewards:torch.Tensor, next_states:torch.Tensor, terminals:torch.Tensor):
        this_states = transform_states(this_states)
        next_states = transform_states(next_states)
        rewards = transform_rewards(rewards, terminals)
        self.buffer.push(this_states, actions, next_states, rewards, terminals)

    def __call__(self, this_states:torch.Tensor, pq:bool=False) -> torch.Tensor:
        this_states = transform_states(this_states)
        q = self.main(this_states)
        p = torch.softmax(q / self.temperature, dim=1)
        indices = torch.multinomial(p, num_samples=1).squeeze(1)
        actions = torch.stack([indices // 2, indices % 2], dim=1).to(torch.bool)
        if pq:
            return actions, q, p
        else:
            return actions
