import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import networks

torch.set_grad_enabled(False)

class Agent:
    def __init__(self, policy_params:dict, optimizer_params:dict, batch_size:int, discount:float, polyak:float, device:str="cpu", path:str=None):
        self.batch_size = batch_size
        self.discount = discount
        self.polyak = polyak
        self.device = device

        self.policy = networks.Policy(**policy_params).to(self.device)
        self.target = networks.Policy(**policy_params).to(self.device)
        self.target.requires_grad_(False)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.AdamW(self.policy.parameters(), **optimizer_params)
        self.history = {"value": [], "error": []}

        if path is not None:
            self.load(path)

    def save(self, path:str):
        state_dict = {
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history
        }
        torch.save(state_dict, path)

    def load(self, path:str):
        state_dict = torch.load(path, weights_only=False, map_location="cpu")
        self.policy.load_state_dict(state_dict["policy"])
        self.target.load_state_dict(state_dict["target"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.history = state_dict["history"]

    def plot(self, path:str):
        for key, val in self.history.items():
            plt.figure(figsize=(5, 5), dpi=300)
            plt.plot(val)
            plt.xlabel("step")
            plt.ylabel(key)
            plt.tight_layout()
            plt.savefig(f"{path}/{key}.png")
            plt.close()

    def step(self, this_states:torch.Tensor, actions:torch.Tensor, next_states:torch.Tensor, rewards:torch.Tensor, terminals:torch.Tensor, weights:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(this_states) == len(actions) == len(next_states) == len(rewards) == len(terminals) == len(weights)

        this_states = this_states.to(self.device, torch.float)
        actions = actions.to(self.device, torch.long)
        next_states = next_states.to(self.device, torch.float)
        rewards = rewards.to(self.device, torch.float)
        terminals = terminals.to(self.device, torch.bool)
        weights = weights.to(self.device, torch.float)
        
        v = self.target.evaluate(next_states)
        y = rewards + self.discount * v * (~terminals)
        
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

        values = torch.min(q1, q2)
        self.history["value"].append(values.mean().item())
        self.history["error"].append(errors.mean().item())
        
        return errors
