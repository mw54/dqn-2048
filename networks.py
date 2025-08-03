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
        x = x.flatten(1, 2)
        x = self.model(x)
        return x
