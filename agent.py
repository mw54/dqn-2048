import torch
import torch.optim as optim
import networks

DEVICE = torch.device("mps")
torch.set_grad_enabled(False)

class Agent:
    def __init__(self, policy_params:dict, optimizer_params:dict, batch_size:int, discount:float, polyak:float, path:str=None):
        self.policy = networks.Policy(**policy_params).to(DEVICE)
        self.target = networks.Policy(**policy_params).to(DEVICE)
        self.optimizer = optim.AdamW(self.policy.parameters(), **optimizer_params)
        self.target.requires_grad_(False)
        self.target.load_state_dict(self.policy.state_dict())

        self.batch_size = batch_size
        self.discount = discount
        self.polyak = polyak

        if path is not None:
            self.load(path)

    def save(self, path:str):
        state_dict = {
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(state_dict, path)

    def load(self, path:str):
        state_dict = torch.load(path, weights_only=False, map_location="cpu")
        self.policy.load_state_dict(state_dict["policy"])
        self.target.load_state_dict(state_dict["target"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def step(self, this_states:torch.Tensor, actions:torch.Tensor, next_states:torch.Tensor, rewards:torch.Tensor, terminals:torch.Tensor, weights:torch.Tensor) -> tuple[torch.Tensor]:
        assert len(this_states) == len(actions) == len(next_states) == len(rewards) == len(terminals) == len(weights)

        this_states = this_states.to(DEVICE, torch.float)
        actions = actions.to(DEVICE, torch.long)
        next_states = next_states.to(DEVICE, torch.float)
        rewards = rewards.to(DEVICE, torch.float)
        terminals = terminals.to(DEVICE, torch.bool)
        weights = weights.to(DEVICE, torch.float)
        
        v, h = self.target.evaluate(next_states)
        y = rewards + self.discount * (v + h) * (~terminals)
        
        with torch.enable_grad():
            self.optimizer.zero_grad()
            q1, q2 = self.policy(this_states)
            q1 = torch.gather(q1, dim=1, index=actions[:,None])[:,0]
            q2 = torch.gather(q2, dim=1, index=actions[:,None])[:,0]
            errors = (y - q1)**2 + (y - q2)**2
            loss = torch.dot(weights, errors)
            loss.backward()
            self.optimizer.step()

        for target_param, policy_param in zip(self.target.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.polyak * policy_param.data + (1 - self.polyak) * target_param.data)

        return errors, q1, q2, h
