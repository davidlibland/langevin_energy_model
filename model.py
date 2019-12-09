from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mcmc.abstract import MCSampler
from mcmc.langevin import LangevinSampler
from mcmc.mala import MALASampler
from utils.resnet import BasicBlock as BasicResnetBlock
from utils.resnet import Swish

LANG_INIT_NS = 1


class BaseEnergyModel(nn.Module):
    def __init__(self, num_features, prior_scale=LANG_INIT_NS, grad_max=100, mc_dynamics=None):
        super().__init__()
        self.prior_scale = prior_scale
        self.grad_max = grad_max
        self.num_features = num_features
        self._log_z_prior = self.num_features*(0.5*np.log(2*np.pi) + np.log(prior_scale))
        if mc_dynamics is None:
            mc_dynamics = MALASampler(lr=0.1)
        self.sampler = mc_dynamics

    def sample_from_prior(self, size: int, device=None):
        """Returns samples from the prior."""
        return torch.randn(size, self.num_features, device=device)*self.prior_scale

    def sample_fantasy(self, x: torch.Tensor=None, num_mc_steps=100, beta=None,
                       num_samples=None, mc_dynamics: MCSampler=None):
        """
        Sample fantasy particles.

        Args:
            x: An initial seed for the sampler. If None, then x will be sampled
                from the prior.
            num_mc_steps: The number of MC steps to take.
            beta: The inverse temperature.
                Must be a float or broadcastable to x. Defaults to 1.
            num_samples: If x is not provided, this many samples will be taken
                from the prior.
            mc_dynamics: The type of dynamincs to use. Defaults to langevin.
            **kwargs: Any addition kwargs to be passed to the dynamics

        Returns:
            Samples.
        """
        if x is None:
            assert isinstance(num_samples, int), \
                "If x is not provided, then the number of samples must " \
                "be specified."
            x = self.sample_from_prior(num_samples)
        if mc_dynamics is None:
            mc_dynamics = self.sampler
        for _ in range(num_mc_steps):
            x = mc_dynamics(self, x, beta=beta)
        return x

    def energy(self, x):
        """Override this in subclasses"""
        raise NotImplementedError

    def forward(self, *input: Any, **kwargs: Any):
        """A default forward call which incorporates the inverse temperature
        and prior."""
        x = input[0]
        prior_energy = torch.sum(((x/self.prior_scale)**2)/2, dim=1, keepdim=True)\
                       + self._log_z_prior
        h = self.energy(x)

        beta = kwargs.get("beta")
        if beta is None:
            beta = torch.ones_like(h)

        return beta*h + prior_energy


class SimpleEnergyModel(BaseEnergyModel):
    def __init__(self, num_inputs, num_layers, num_units, prior_scale=LANG_INIT_NS):
        super().__init__(num_features=num_inputs, prior_scale=prior_scale)
        input_layer = nn.utils.spectral_norm(nn.Linear(num_inputs, num_units))
        self.internal_layers = nn.ModuleList([input_layer])
        for _ in range(num_layers-2):
            layer = nn.utils.spectral_norm(nn.Linear(num_units, num_units))
            self.internal_layers.append(layer)
        output_layer = nn.Linear(num_units, 1)
        self.internal_layers.append(output_layer)
        # ToDo: Add weight initialization

    def energy(self, x, **kwargs):
        for layer in self.internal_layers[:-1]:
            x = layer(x)
            x = F.leaky_relu(x)
        x = self.internal_layers[-1](x)
        return x


class ConvEnergyModel(BaseEnergyModel):
    def __init__(self, input_shape, num_layers=3, num_units=25, prior_scale=LANG_INIT_NS):
        c, h, w = input_shape
        num_features = c*h*w
        super().__init__(num_features=num_features, prior_scale=prior_scale)
        self.input_shape = input_shape
        self.internal_layers = nn.ModuleList()
        in_channels = c
        kernel_size = (3, 3)
        for _ in range(num_layers-1):
            layer = nn.utils.spectral_norm(nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_units,
                kernel_size=kernel_size
            ))
            w -= kernel_size[1] - 1
            h -= kernel_size[0] - 1
            in_channels = num_units
            self.internal_layers.append(layer)
        self.dense_size=w*h*num_units
        dense_layer = nn.Linear(self.dense_size, 1)
        self.internal_layers.append(dense_layer)
        # ToDo: Add weight initialization

    def energy(self, x):
        n = x.shape[0]
        x = torch.reshape(x, (n, )+self.input_shape)
        for layer in self.internal_layers[:-1]:
            x = layer(x)
            x = F.leaky_relu(x)
        x = torch.reshape(x, (n, self.dense_size))
        x = self.internal_layers[-1](x)
        return x


class ResnetEnergyModel(BaseEnergyModel):
    def __init__(self, input_shape, num_layers=3, num_resnets=2, num_units=25, prior_scale=LANG_INIT_NS):
        c, h, w = input_shape
        num_features = c*h*w
        super().__init__(num_features=num_features, prior_scale=prior_scale)
        self.input_shape = input_shape
        self.internal_layers = nn.ModuleList()
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

            # down_sample = BasicResnetBlock(
            #     in_channels=in_channels,
            #     out_channels=num_units,
            #     norm_layer=None,
            #     stride=2
            # )
            avg_pool = nn.AvgPool2d(kernel_size=2, padding=0)
            # w -= kernel_size[1] - 1
            # h -= kernel_size[0] - 1
            w //= 2
            h //= 2
            self.internal_layers.append(avg_pool)
        self.dense_size=num_units
        dense_layer = nn.Linear(self.dense_size, 1)
        self.internal_layers.append(dense_layer)
        # ToDo: Add weight initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def energy(self, x):
        n = x.shape[0]
        x = torch.reshape(x, (n, )+self.input_shape)
        ones = torch.ones_like(x)
        x = torch.cat([ones, x], dim=1) # add a channel of 1s to distinguish padding.
        for layer in self.internal_layers[:-1]:
            x = layer(x)
            # ToDo: Consider changing this to swish:
            x = F.leaky_relu(x)
        x = torch.sum(x, dim=[2, 3])  # spatial sum
        x = self.internal_layers[-1](x)
        return x
