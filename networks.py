import torch
import torch.nn as nn
import torch.nn.functional as F

class Value(nn.Module):
    def __init__(self, input_channels:int, model_channels:int, output_channels:int, seq_len:int, num_heads:int, num_layers:int, dropout:float):
        super(Value, self).__init__()
        self.embed = nn.Linear(input_channels, model_channels, bias=False)
        self.encode = nn.Parameter(torch.randn(1, seq_len, model_channels))
        self.mlp = nn.Sequential(
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, 4 * model_channels),
            nn.GELU(),
            nn.Linear(4 * model_channels, model_channels)
        )

        layer = nn.TransformerEncoderLayer(model_channels, num_heads, 4 * model_channels, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

        self.gate = nn.Sequential(
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, 4 * model_channels),
            nn.GELU(),
            nn.Linear(4 * model_channels, num_heads),
            nn.Softmax(dim=1)
        )
        self.value = nn.Sequential(
            nn.LayerNorm(num_heads * model_channels),
            nn.Linear(num_heads * model_channels, 4 * model_channels),
            nn.GELU(),
            nn.Linear(4 * model_channels, output_channels)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.embed(x) + self.encode)
        x = self.transformer(x)
        gate = self.gate(x).permute(0, 2, 1)
        x = torch.sum(gate[:,:,:,None] * x[:,None,:,:], dim=2).flatten(1, 2)
        x = self.value(x)
        return x
    
class Policy(nn.Module):
    def __init__(self, model_channels:int, seq_len:int, num_heads:int, num_layers:int, dropout:int, temperature:float):
        super(Policy, self).__init__()
        self.q1 = Value(18, model_channels, 4, seq_len, num_heads, num_layers, dropout)
        self.q2 = Value(18, model_channels, 4, seq_len, num_heads, num_layers, dropout)
        self.temperature = temperature
        self.seq_len = seq_len

    def embed(self, x:torch.Tensor) -> torch.Tensor:
        x = x.flatten(1, 2)
        x = F.one_hot(x.masked_fill(x == 0, 1.0).log2().long(), num_classes=18).to(torch.float)
        return x

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(x)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2
    
    def eval(self, x:torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        q1 = self.q1(x)
        q2 = self.q2(x)
        qs = torch.min(q1, q2)
        v = torch.max(qs, dim=1).values
        return v
    
    def act(self, x:torch.Tensor, stochastic=True) -> torch.Tensor:
        x = self.embed(x)
        q1 = self.q1(x)
        q2 = self.q2(x)
        qs = torch.min(q1, q2)
        ps = torch.softmax(qs / self.temperature, dim=1)
        if stochastic:
            actions = torch.multinomial(ps, num_samples=1)[:,0]
        else:
            actions = torch.argmax(ps, dim=1)
        return actions
