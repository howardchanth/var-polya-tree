import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class sigmoid_projection(nn.Module):
    def __init__(self):
        super(sigmoid_projection, self).__init__()
    def forward(self, x):
        return 1 / (1 + torch.exp(-x))


class MLPS(nn.Sequential):
    def __init__(self, input_size, hidden_sizes, output_size, activation="tanh", flatten=False, bias=True):
        super(MLPS, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        if output_size is not None:
            self.output_size = output_size
        else:
            self.output_size = 1

        # Set activation function
        if activation == "relu":
            act = nn.ReLU
        elif activation == "tanh":
            act = nn.Tanh
        else:
            raise ValueError('invalid activation')

        if flatten:
            self.add_module('flatten', nn.Flatten())

        if len(hidden_sizes) == 0:
            # Linear Model
            self.add_module('lin_layer', nn.Linear(self.input_size, self.output_size, bias=bias))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i+1}', nn.Linear(in_size, out_size, bias=bias))
                self.add_module(f'{activation}{i+1}', act())
            self.add_module('out_layer', nn.Linear(hidden_sizes[-1], self.output_size, bias=bias))
            self.add_module('out_activation', sigmoid_projection())

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation="tanh", **kwargs):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        if output_size is not None:
            self.output_size = output_size
        else:
            self.output_size = 1

        # Set activation function
        if activation == "relu":
            self.act = torch.nn.ReLU
        elif activation == "tanh":
            self.act = torch.tanh
        elif activation == "sigmoid":
            self.act = torch.sigmoid
        elif activation == "selu":
            self.act = F.elu

        # Define layers
        if len(hidden_sizes) == 0:
            # Linear model
            self.hidden_layers = []
            self.output_layer = nn.Linear(self.input_size, self.output_size)
        else:
            # Neural network
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size
                                                in in_outs])
            self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            out = self.act(layer(out))
        z = self.output_layer(out)
        return z.flatten() if self.output_size == 1 else z

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim, delta, w_star):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.input_dim = len(w_star)
        self.delta = delta
        with torch.no_grad():
            if isinstance(w_star, np.ndarray):
                self.w_star = torch.from_numpy(w_star).float()
            else:
                self.w_star = w_star

    def forward(self, x):
        z = self.linear(x)
        return z


class LeNet5(nn.Module):
    def __init__(self, output_size=10):
        super().__init__()
        self.output_size = output_size

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.act3 = nn.Tanh()

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1 * 1 * 120, 84)
        self.act4 = nn.Tanh()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # input 1x28x28, output 6x28x28
        x = self.act1(self.conv1(x))
        # input 6x28x28, output 6x14x14
        x = self.pool1(x)
        # input 6x14x14, output 16x10x10
        x = self.act2(self.conv2(x))
        # input 16x10x10, output 16x5x5
        x = self.pool2(x)
        # input 16x5x5, output 120x1x1
        x = self.act3(self.conv3(x))
        # input 120x1x1, output 84
        x = self.act4(self.fc1(self.flat(x)))
        # input 84, output 10
        x = self.fc2(x)
        return x

class LeNet5_3channels(nn.Module):
    def __init__(self):
        super(LeNet5_3channels, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x