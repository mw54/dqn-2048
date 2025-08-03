import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation="ReLU"):
        super(MLP, self).__init__()
        activation = getattr(nn, activation)
        layers = list()
        in_size = input_size
        for out_size in hidden_sizes:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(activation())
            in_size = out_size
        layers.append(nn.Linear(in_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x

class DuelingMLP(nn.Module):
    def __init__(self, input_size, embed_hidden, embed_size, value_hidden, advantage_hidden, output_size, activation="ReLU"):
        super(DuelingMLP, self).__init__()
        self.embed = MLP(input_size, embed_hidden, embed_size, activation)
        self.value = MLP(embed_size, value_hidden, 1, activation)
        self.advantage = MLP(embed_size, advantage_hidden, output_size, activation)
        self.activation = getattr(nn, activation)()

    def forward(self, x):
        x = torch.flatten(x, 1, 2)
        x = self.activation(self.embed(x))
        value = self.value(x)
        advantage = self.advantage(x)
        x = value + advantage - torch.mean(advantage, dim=1, keepdim=True)
        return x
    