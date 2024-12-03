import torch.nn as nn
import torch.nn.functional as F
import torch


def sigmoid_projection(v):
    return 1 / (1 + torch.exp(-v))

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_size[0]))

        for i in range(len(hidden_size) - 1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))

        self.layers.append(nn.Linear(hidden_size[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = sigmoid_projection(x)
        return x

