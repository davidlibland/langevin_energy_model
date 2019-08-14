from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseEnergyModel(nn.Module):
    def sample_fantasy(self, x: torch.Tensor, num_mc_steps=100, lr=1e-3, sigma=1):
        for _ in range(num_mc_steps):
            x = self.fantasy_step(x, lr=lr, sigma=sigma)
        return x

    def fantasy_step(self, x: torch.Tensor, lr=1e-3, sigma=1):
        noise_scale = sigma*torch.sqrt(torch.as_tensor(lr*2))
        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.data.zero_()
        y = self(x).sum()
        y.backward()
        grad_x = x.grad
        result = x - lr*grad_x+noise_scale*torch.randn_like(x)
        return result.detach()


class SimpleEnergyModel(BaseEnergyModel):
    def __init__(self, num_inputs, num_layers, num_units):
        super().__init__()
        input_layer = nn.Linear(num_inputs, num_units)
        self.internal_layers = nn.ModuleList([input_layer])
        for _ in range(num_layers-2):
            layer = nn.Linear(num_units, num_units)
            self.internal_layers.append(layer)
        output_layer = nn.Linear(num_units, 1)
        self.internal_layers.append(output_layer)
        # ToDo: Add weight initialization

    def forward(self, x):
        for layer in self.internal_layers[:-1]:
            x = layer(x)
            x = F.leaky_relu(x)
        x = self.internal_layers[-1](x)
        # ToDo: Make strength of quadratic term tunable
        return x + torch.sum(x**2, axis=1, keepdim=True)


class ConvEnergyModel(BaseEnergyModel):
    def __init__(self, input_shape, num_layers=3, num_units=25):
        super().__init__()
        self.input_shape = input_shape
        self.internal_layers = nn.ModuleList()
        c, h, w = input_shape
        in_channels = c
        kernel_size = (3, 3)
        for _ in range(num_layers-1):
            layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_units,
                kernel_size=kernel_size
            )
            w -= kernel_size[1] - 1
            h -= kernel_size[0] - 1
            in_channels = num_units
            self.internal_layers.append(layer)
        self.dense_size=w*h*num_units
        dense_layer = nn.Linear(self.dense_size, 1)
        self.internal_layers.append(dense_layer)
        # ToDo: Add weight initialization

    def forward(self, x):
        x = torch.reshape(x, (-1, )+self.input_shape)
        for layer in self.internal_layers[:-1]:
            x = layer(x)
            x = F.leaky_relu(x)
        x = torch.reshape(x, (-1, self.dense_size))
        x = self.internal_layers[-1](x)
        # ToDo: Make strength of quadratic term tunable
        return x + torch.sum(x**2, axis=1, keepdim=True)