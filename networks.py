import torch
import torch.nn as nn

class Value(nn.Module):
    def __init__(self, model_channels:int, hidden_channels:int, output_channels:int, num_cells:int, num_heads:int, num_layers:int, dropout:float):
        super(Value, self).__init__()
        layer = nn.TransformerEncoderLayer(model_channels, num_heads, 4 * model_channels, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.mlp = nn.Sequential(
            nn.Linear(num_cells * model_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, output_channels)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.transformer(x)
        x = x.flatten(1, 2)
        x = self.mlp(x)
        return x
    
class Policy(nn.Module):
    def __init__(self, model_channels:int, hidden_channels:int, num_cells:int, num_heads:int, num_layers:int, dropout:int):
        super(Policy, self).__init__()
        self.tiles = nn.Embedding(18, model_channels)
        self.cells = nn.Parameter(torch.randn(1, num_cells, model_channels))
        self.value1 = Value(model_channels, hidden_channels, 4, num_cells, num_heads, num_layers, dropout)
        self.value2 = Value(model_channels, hidden_channels, 4, num_cells, num_heads, num_layers, dropout)

    def tokenize(self, x:torch.Tensor) -> torch.Tensor:
        x = x.masked_fill(x == 0, 1.0).log2().long().flatten(1, 2)
        x = self.tiles(x) + self.cells
        return x

    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokenize(x)
        q1 = self.value1(x)
        q2 = self.value2(x)
        return q1, q2
    
    def evaluate(self, x:torch.Tensor) -> torch.Tensor:
        x = self.tokenize(x)
        q1 = self.value1(x)
        q2 = self.value2(x)
        qs = torch.min(q1, q2)
        vs = torch.max(qs, dim=1).values
        return vs
    
    def act(self, x:torch.Tensor, temperature:float=None) -> torch.Tensor:
        x = self.tokenize(x)
        q1 = self.value1(x)
        q2 = self.value2(x)
        qs = torch.min(q1, q2)
        if temperature is not None:
            ps = torch.softmax(qs / temperature, dim=1)
            actions = torch.multinomial(ps, num_samples=1)[:,0]
        else:
            actions = torch.argmax(qs, dim=1)
        return actions
