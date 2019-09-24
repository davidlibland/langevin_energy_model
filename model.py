from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.resnet import BasicBlock as BasicResnetBlock
from utils.resnet import Swish

LANG_INIT_NS = 1


class BaseEnergyModel(nn.Module):
    def __init__(self, prior_scale=LANG_INIT_NS, grad_max=100):
        super().__init__()
        self.prior_scale = prior_scale
        self.grad_max = grad_max

    def sample_fantasy(self, x: torch.Tensor, num_mc_steps=100, lr=1e-3, sigma=1):
        for _ in range(num_mc_steps):
            x = self.fantasy_step(x, lr=lr, sigma=sigma)
        return x

    def fantasy_step(self, x: torch.Tensor, lr=1e-3, sigma=1):
        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.data.zero_()
        y = self(x).sum()
        y.backward()
        grad_x = x.grad

        # Hack to keep gradients in control:
        lr = lr/max(1, grad_x.abs().max())

        noise_scale = sigma*torch.sqrt(torch.as_tensor(lr*2))
        result = x - lr*grad_x+noise_scale*torch.randn_like(x)
        return result.detach()


class SimpleEnergyModel(BaseEnergyModel):
    def __init__(self, num_inputs, num_layers, num_units, prior_scale=LANG_INIT_NS):
        super().__init__(prior_scale=prior_scale)
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
        return x + torch.sum(((x/self.prior_scale)**2)/2, dim=1, keepdim=True)


class ConvEnergyModel(BaseEnergyModel):
    def __init__(self, input_shape, num_layers=3, num_units=25, prior_scale=LANG_INIT_NS):
        super().__init__(prior_scale=prior_scale)
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
        n = x.shape[0]
        x = torch.reshape(x, (n, )+self.input_shape)
        for layer in self.internal_layers[:-1]:
            x = layer(x)
            x = F.leaky_relu(x)
        x = torch.reshape(x, (n, self.dense_size))
        x = self.internal_layers[-1](x)
        return x + torch.sum(((x/self.prior_scale)**2)/2, dim=1, keepdim=True)


class ResnetEnergyModel(BaseEnergyModel):
    def __init__(self, input_shape, num_layers=3, num_resnets=2, num_units=25, prior_scale=LANG_INIT_NS):
        super().__init__(prior_scale=prior_scale)
        self.input_shape = input_shape
        self.internal_layers = nn.ModuleList()
        c, h, w = input_shape
        in_channels = c + 1
        kernel_size = (3, 3)
        for _ in range(num_layers-1):
            for _ in range(num_resnets):
                res_net_layer = BasicResnetBlock(
                    in_channels=in_channels,
                    out_channels=num_units,
                    norm_layer=None,
                    activation=Swish,
                    spectral_norm=True,
                )
                self.internal_layers.append(res_net_layer)
                in_channels = num_units

            down_sample = BasicResnetBlock(
                in_channels=in_channels,
                out_channels=num_units,
                norm_layer=None,
                stride=2
            )
            # maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
            # w -= kernel_size[1] - 1
            # h -= kernel_size[0] - 1
            w //= 2
            h //= 2
            self.internal_layers.append(down_sample)
        self.dense_size=w*h*num_units
        dense_layer = nn.Linear(self.dense_size, 1)
        self.internal_layers.append(dense_layer)
        # ToDo: Add weight initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n = x.shape[0]
        x = torch.reshape(x, (n, )+self.input_shape)
        ones = torch.ones_like(x)
        x = torch.cat([ones, x], dim=1) # add a channel of 1s to distinguish padding.
        for layer in self.internal_layers[:-1]:
            x = layer(x)
            x = F.leaky_relu(x)
        x = torch.reshape(x, (n, self.dense_size))
        x = self.internal_layers[-1](x)
        return x + torch.sum(((x/self.prior_scale)**2)/2, dim=1, keepdim=True)
