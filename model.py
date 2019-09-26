from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.resnet import BasicBlock as BasicResnetBlock
from utils.resnet import Swish

LANG_INIT_NS = 1


class BaseEnergyModel(nn.Module):
    def __init__(self, num_features, prior_scale=LANG_INIT_NS, grad_max=100):
        super().__init__()
        self.prior_scale = prior_scale
        self.grad_max = grad_max
        self.num_features = num_features
        self._log_z_prior = self.num_features*(0.5*np.log(2*np.pi) + np.log(prior_scale))

    def sample_from_prior(self, size: int, device=None):
        """Returns samples from the prior."""
        return torch.randn(size, self.num_features, device=device)*self.prior_scale

    def sample_fantasy(self, x: torch.Tensor=None, num_mc_steps=100, beta=None,
                       num_samples=None, mc_dynamics="mala", mc_kwargs=None):
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
        mc_transition = {
           "langevin":  self.langavin_fantasy_step,
            "mala": self.mala_fantasy_step
        }.get(mc_dynamics, self.langavin_fantasy_step)
        if mc_kwargs is None:
            mc_kwargs = {}
        for _ in range(num_mc_steps):
            x, new_kwargs = mc_transition(x, beta=beta, **mc_kwargs)
            for key in new_kwargs:
                mc_kwargs[key] = new_kwargs[key]
        return x

    def langavin_fantasy_step(self, x: torch.Tensor, beta=None, lr=1e-3, **kwargs):
        """Perform a single langevin MC update."""
        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.data.zero_()
        y = self(x, beta=beta).sum()
        y.backward()
        grad_x = x.grad

        # Hack to keep gradients in control:
        lr = lr/max(1, grad_x.abs().max())

        noise_scale = torch.sqrt(torch.as_tensor(lr*2))
        result = x - lr*grad_x+noise_scale*torch.randn_like(x)
        return result.detach(), {}

    def mala_fantasy_step(self, x: torch.Tensor, beta=None, lr=1e-3, **kwargs):
        """Perform a single langevin MC update."""
        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.data.zero_()
        y = self(x, beta=beta)
        y.sum().backward()
        grad_x = x.grad

        lr_initial = lr
        # Hack to keep gradients in control:
        lr = lr/max(1, grad_x.abs().max())

        noise_scale = torch.sqrt(torch.as_tensor(lr*2))
        x_det = (x - lr*grad_x).detach()
        noise_f = noise_scale*torch.randn_like(x)
        x_ = x_det+noise_f

        log_q_x_x = -(noise_f**2).sum(dim=1, keepdim=True)/(4*lr)

        x_.requires_grad_(True)
        if x_.grad is not None:
            x_.grad.data.zero_()
        y_ = self(x_, beta=beta)
        y_.sum().backward()
        grad_x_ = x_.grad

        eps = ((x - x_ + lr * grad_x_) ** 2).sum(dim=1, keepdim=True)
        log_q_xx_ = -eps/(4*lr)

        log_alpha = y - y_ + log_q_xx_ - log_q_x_x
        alpha = torch.exp(torch.clamp_max(log_alpha, 0))
        mask = torch.rand(x.shape[0], 1) < alpha
        # adjust the learning rate based on the acceptance ratio:
        acceptance_ratio = torch.mean(mask.float()).float()
        if acceptance_ratio.float() < .4:
            lr = lr_initial / 1.1
        elif acceptance_ratio.float() > .7:
            lr = lr_initial * 1.1
        ac_r = float(0.1*acceptance_ratio+.9*kwargs.get("acceptance_prob", 0.5))
        return torch.where(mask, x_, x).detach(), {"lr": lr, "acceptance_prob": ac_r}

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
        input_layer = nn.Linear(num_inputs, num_units)
        self.internal_layers = nn.ModuleList([input_layer])
        for _ in range(num_layers-2):
            layer = nn.Linear(num_units, num_units)
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

    def energy(self, x):
        n = x.shape[0]
        x = torch.reshape(x, (n, )+self.input_shape)
        ones = torch.ones_like(x)
        x = torch.cat([ones, x], dim=1) # add a channel of 1s to distinguish padding.
        for layer in self.internal_layers[:-1]:
            x = layer(x)
            x = F.leaky_relu(x)
        x = torch.reshape(x, (n, self.dense_size))
        x = self.internal_layers[-1](x)
        return x
