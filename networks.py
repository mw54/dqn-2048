import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, input_channels:int, model_channels:int, output_channels:int, seq_len:int, num_heads:int, num_layers:int, dropout:float):
        super(QNet, self).__init__()
        self.embed = nn.Linear(input_channels, model_channels, bias=False)
        self.encode = nn.Linear(seq_len, model_channels, bias=False)

        layer = nn.TransformerEncoderLayer(model_channels, num_heads, 4 * model_channels, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

        self.output = nn.Sequential(
            nn.Linear(model_channels, 2 * model_channels // seq_len),
            nn.GELU()
        )
        self.value = nn.Sequential(
            nn.Linear(2 * model_channels, 4 * model_channels),
            nn.GELU(),
            nn.Linear(4 * model_channels, output_channels)
        )

    def forward(self, x:torch.Tensor, p:torch.Tensor):
        x = self.embed(x) + self.encode(p)
        x = self.transformer(x)
        x = self.output(x).flatten(1, -1)
        x = self.value(x)
        return x
    
class Policy(nn.Module):
    def __init__(self, model_channels:int, seq_len:int, num_heads:int, num_layers:int, dropout:int, temperature:float):
        super(Policy, self).__init__()
        self.q1 = QNet(18, model_channels, 4, seq_len, num_heads, num_layers, dropout)
        self.q2 = QNet(18, model_channels, 4, seq_len, num_heads, num_layers, dropout)
        self.temperature = temperature
        self.seq_len = seq_len

    def embed(self, x:torch.Tensor):
        x = x.flatten(1, 2)
        x = F.one_hot(x.masked_fill(x == 0, 1.0).log2().long(), num_classes=18).to(torch.float)
        p = torch.eye(self.seq_len, x.size(1), device=x.device, dtype=torch.float)[None,:,:]
        return x, p

    def forward(self, x:torch.Tensor):
        x, p = self.embed(x)
        q1 = self.q1(x, p)
        q2 = self.q2(x, p)
        return q1, q2
    
    def evaluate(self, x:torch.Tensor) -> tuple[torch.Tensor]:
        x, p = self.embed(x)
        q1 = self.q1(x, p)
        q2 = self.q2(x, p)
        qs = torch.min(q1, q2)
        ps = torch.softmax(qs / self.temperature, dim=1)
        v = torch.sum(qs * ps, dim=1)
        h = torch.sum(-ps * ps.log(), dim=1)
        return v, h
    
    def act(self, x:torch.Tensor, stochastic=True) -> torch.Tensor:
        x, p = self.embed(x)
        q1 = self.q1(x, p)
        q2 = self.q2(x, p)
        qs = torch.min(q1, q2)
        ps = torch.softmax(qs / self.temperature, dim=1)
        if stochastic:
            actions = torch.multinomial(ps, num_samples=1)[:,0]
        else:
            actions = torch.argmax(ps, dim=1)
        return actions
