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
        x = torch.flatten(x, 1, -1)
        x = self.model(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, input_channel, input_size, hidden_channels, hidden_sizes, output_size, activation):
        super(ConvNet, self).__init__()
        activation = getattr(nn, activation)
        in_size = input_channel
        embed = list()
        for out_size in hidden_channels:
            embed.append(nn.Conv2d(in_size, out_size, 3, 1, 1))
            embed.append(activation())
            in_size = out_size
        self.embed = nn.Sequential(*embed)

        in_size = in_size * input_size
        unembed = list()
        for out_size in hidden_sizes:
            unembed.append(nn.Linear(in_size, out_size))
            unembed.append(activation())
            in_size = out_size
        unembed.append(nn.Linear(in_size, output_size))
        self.unembed = nn.Sequential(*unembed)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embed(x)
        x = torch.flatten(x, 1, -1)
        x = self.unembed(x)
        return x
