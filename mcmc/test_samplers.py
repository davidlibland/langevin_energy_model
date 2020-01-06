"""
This file adds some MC tests for samplers.
"""
import pytest
import torch
import matplotlib.pyplot as plt
import numpy as np

import mcmc.langevin
import mcmc.mala
import mcmc.simulated_tempering
import model


class NormalNet(model.BaseEnergyModel):
    def __init__(self, scale=1, mc_dynamics=None):
        num_features=1
        super().__init__(num_features=num_features, mc_dynamics=mc_dynamics, prior_scale=1)
        self.scale = scale

    def energy(self, x):
        energy = torch.sum(((x/self.scale)**2)/2, dim=1, keepdim=True)
        return energy

    def variance(self, beta=1):
        """Computes the variance at a given beta."""
        inv_var = beta*0.5/self.scale**2 + 0.5/self.prior_scale**2
        return inv_var


class MixtureNormalNet(model.BaseEnergyModel):
    def __init__(self, locs=(-6, 6), scales=(1, 1), mc_dynamics=None):
        num_features=1
        super().__init__(num_features=num_features, mc_dynamics=mc_dynamics, prior_scale=10)
        self.locs = locs
        self.scales = scales

    def energy(self, x):
        energies = [
            torch.sum((((x-loc)/scale)**2)/2, dim=1, keepdim=True)
            for loc, scale in zip(self.locs, self.scales)
        ]
        return -torch.logsumexp(-torch.stack(energies, dim=1), dim=1)


@pytest.mark.parametrize("name, fsampler", [
    ("langevin", lambda: mcmc.langevin.LangevinSampler(lr=0.1)),
    ("mala", lambda: mcmc.mala.MALASampler(lr=0.1)),
    ("tempered langevin", lambda: mcmc.simulated_tempering.SimulatedTempering(mc_dynamics=mcmc.langevin.LangevinSampler(lr=0.1))),
    ("tempered mala", lambda: mcmc.simulated_tempering.SimulatedTempering(mc_dynamics=mcmc.mala.MALASampler(lr=0.1)))
])
def test_normal_stats(name, fsampler, mean_tolerance=0.05, std_tolerance=0.05, num_steps=100, num_samples=1000):
    """Tests that samplers derive correct normal stats."""
    net = NormalNet(mc_dynamics=fsampler())
    samples = []
    for _ in range(num_steps):
        x = net.sample_fantasy(num_mc_steps=1, num_samples=num_samples)
        samples.append(x)
    all_x = torch.stack(samples[num_steps//2:])
    mean = torch.mean(all_x)
    std = torch.sqrt(torch.sum(all_x**2)/all_x.numel())
    assert abs(mean) < mean_tolerance, f"Mean via {name} was not accurate"
    expected = np.sqrt(net.variance())
    assert abs(std - expected) < std_tolerance, f"Standard Deviation via {name} was not accurate"


test_samplers = {
    "langevin": lambda: mcmc.langevin.LangevinSampler(lr=0.1),
    "mala": lambda: mcmc.mala.MALASampler(lr=0.1),
    "tempered langevin": lambda: mcmc.simulated_tempering.SimulatedTempering(mc_dynamics=mcmc.langevin.LangevinSampler(lr=0.1)),
    "tempered mala": lambda: mcmc.simulated_tempering.SimulatedTempering(mc_dynamics=mcmc.mala.MALASampler(lr=0.1))
}

def plot_samplers(fsamplers=test_samplers, num_steps=100, net_factory=MixtureNormalNet):
    """Plots samplers to compare them visually."""
    f, ax = plt.subplots(1, 2)
    vals = {}
    for name, fsampler in fsamplers.items():
        net = net_factory(mc_dynamics=fsampler())
        samples = []
        for _ in range(num_steps):
            x = net.sample_fantasy(num_mc_steps=1, num_samples=1)
            samples.append(x)
        all_x = torch.stack(samples[num_steps//2:])
        mean = torch.mean(all_x)
        var = torch.var(all_x)
        print(f"{name}:\n"
              f" - mean: {mean}\n"
              f" - var: {var}")
        x_np = all_x.numpy().flatten()
        vals[name] = x_np
        ax[0].scatter(np.arange(x_np.size), x_np, label=name, alpha=0.75)
    labels = list(vals.keys())
    xs = np.stack(vals.values(), axis=-1)
    ax[1].hist(xs, label=labels, density=True)
    num_xs = 100
    x_range = np.linspace(xs.min(), xs.max(), num_xs)
    y_range = torch.exp(-net(torch.tensor(x_range.reshape([-1, 1]))))
    cum_y = torch.sum(y_range)
    y_range *= num_xs/(cum_y * (xs.max() - xs.min()))
    ax[1].plot(x_range, y_range)
    f.legend()
    plt.show()


