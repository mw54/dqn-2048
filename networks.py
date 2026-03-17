import torch
import torch.nn as nn
import torch.nn.functional as F

class Value(nn.Module):
    def __init__(self, input_channels:int, model_channels:int, output_channels:int, seq_len:int, num_heads:int, num_layers:int, dropout:float):
        super(Value, self).__init__()
        self.embedding = nn.Linear(input_channels, model_channels, bias=False)
        self.encoding = nn.Parameter(torch.randn(1, seq_len, model_channels))
        self.norm1 = nn.LayerNorm(model_channels)
        self.norm2 = nn.LayerNorm(model_channels)
        self.input = nn.Sequential(
            nn.Linear(model_channels, 4 * model_channels),
            nn.GELU(),
            nn.Linear(4 * model_channels, model_channels)
        )

        layer = nn.TransformerEncoderLayer(model_channels, num_heads, 4 * model_channels, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

        self.gate = nn.Sequential(
            nn.Linear(model_channels, 4 * model_channels),
            nn.GELU(),
            nn.Linear(4 * model_channels, output_channels),
            nn.Softmax(dim=1)
        )
        self.value = nn.Sequential(
            nn.Linear(model_channels, 4 * model_channels),
            nn.GELU(),
            nn.Linear(4 * model_channels, model_channels),
        )
        self.output = nn.Sequential(
            nn.Linear(model_channels, 4 * model_channels),
            nn.GELU(),
            nn.Linear(4 * model_channels, 1)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.input(self.norm1(self.embedding(x)) + self.norm2(self.encoding))
        x = self.transformer(x)
        x = torch.sum(self.value(x)[:,None,:,:] * self.gate(x)[:,:,:,None].permute(0, 2, 1, 3), dim=2)
        x = self.output(x)[:,:,0]
        return x
    
class Policy(nn.Module):
    def __init__(self, model_channels:int, seq_len:int, num_heads:int, num_layers:int, dropout:int):
        super(Policy, self).__init__()
        self.q1 = Value(18, model_channels, 4, seq_len, num_heads, num_layers, dropout)
        self.q2 = Value(18, model_channels, 4, seq_len, num_heads, num_layers, dropout)

    def embed(self, x:torch.Tensor) -> torch.Tensor:
        x = x.flatten(1, 2)
        x = F.one_hot(x.masked_fill(x == 0, 1.0).log2().long(), num_classes=18).to(torch.float)
        return x

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(x)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2
    
    def evaluate(self, x:torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        q1 = self.q1(x)
        q2 = self.q2(x)
        qs = torch.min(q1, q2)
        v = torch.max(qs, dim=1).values
        return v
    
    def act(self, x:torch.Tensor, temperature:float=None) -> torch.Tensor:
        x = self.embed(x)
        q1 = self.q1(x)
        q2 = self.q2(x)
        qs = torch.min(q1, q2)
        if temperature is not None:
            ps = torch.softmax(qs / temperature, dim=1)
            actions = torch.multinomial(ps, num_samples=1)[:,0]
        else:
            actions = torch.argmax(qs, dim=1)
        return actions
