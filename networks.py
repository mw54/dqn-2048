import torch
import torch.nn as nn

class Value(nn.Module):
    def __init__(self, input_channels:int, model_channels:int, output_channels:int, seq_len:int, num_heads:int, num_layers:int, dropout:float):
        super(Value, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_channels, 4 * model_channels),
            nn.GELU(),
            nn.Linear(4 * model_channels, model_channels),
            nn.LayerNorm(model_channels)
        )
        self.encoding = nn.Sequential(
            nn.Linear(seq_len, model_channels),
            nn.LayerNorm(model_channels)
        )
        
        layer = nn.TransformerEncoderLayer(model_channels, num_heads, 4 * model_channels, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

        self.gate = nn.Sequential(
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, 4 * model_channels),
            nn.GELU(),
            nn.Linear(4 * model_channels, output_channels),
            nn.Softmax(dim=1)
        )
        self.value = nn.Sequential(
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, 4 * model_channels),
            nn.GELU(),
            nn.Linear(4 * model_channels, 1),
        )

    def forward(self, x:torch.Tensor, p:torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) + self.encoding(p)
        x = self.transformer(x)
        x = torch.sum(self.gate(x)[:,:,:,None].permute(0, 2, 1, 3) * x[:,None,:,:], dim=2)
        x = self.value(x)[:,:,0]
        return x
    
class Policy(nn.Module):
    def __init__(self, model_channels:int, seq_len:int, num_heads:int, num_layers:int, dropout:int):
        super(Policy, self).__init__()
        self.positions = nn.Buffer(torch.eye(seq_len)[None,:,:])
        self.q1 = Value(1, model_channels, 4, seq_len, num_heads, num_layers, dropout)
        self.q2 = Value(1, model_channels, 4, seq_len, num_heads, num_layers, dropout)

    def embed(self, x:torch.Tensor) -> torch.Tensor:
        x = x.flatten(1, 2)
        x = x.masked_fill(x == 0, 1.0).log2()[:,:,None]
        p = self.positions[:,:x.size(1),:].expand(x.size(0), -1, -1)
        return x, p

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, p = self.embed(x)
        q1 = self.q1(x, p)
        q2 = self.q2(x, p)
        return q1, q2
    
    def evaluate(self, x:torch.Tensor) -> torch.Tensor:
        x, p = self.embed(x)
        q1 = self.q1(x, p)
        q2 = self.q2(x, p)
        qs = torch.min(q1, q2)
        v = torch.max(qs, dim=1).values
        return v
    
    def act(self, x:torch.Tensor, temperature:float=None) -> torch.Tensor:
        x, p = self.embed(x)
        q1 = self.q1(x, p)
        q2 = self.q2(x, p)
        qs = torch.min(q1, q2)
        if temperature is not None:
            ps = torch.softmax(qs / temperature, dim=1)
            actions = torch.multinomial(ps, num_samples=1)[:,0]
        else:
            actions = torch.argmax(qs, dim=1)
        return actions
