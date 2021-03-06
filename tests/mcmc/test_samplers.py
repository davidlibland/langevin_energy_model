"""
This file adds some MC tests for samplers.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import src.mcmc.langevin
import src.mcmc.mala
import src.mcmc.tempered_transitions
from src import model


plt.switch_backend("agg")


class NormalNet(model.BaseEnergyModel):
    def __init__(self, scale=1):
        num_features = 1
        super().__init__(input_shape=(num_features,), prior_scale=1)
        self.scale = scale

    def energy(self, x):
        energy = torch.sum(((x / self.scale) ** 2) / 2, dim=1)
        result = energy.squeeze(-1)
        return result

    def variance(self, beta=1):
        """Computes the variance at a given beta."""
        inv_var = beta / self.scale ** 2 + 1 / self.prior_scale ** 2
        return 1 / inv_var


class MixtureNormalNet(model.BaseEnergyModel):
    def __init__(self, locs=(-6, 6), scales=(1, 1)):
        num_features = 1
        super().__init__(input_shape=(num_features,), prior_scale=10)
        self.locs = locs
        self.scales = scales

    def energy(self, x):
        energies = [
            torch.sum((((x - loc) / scale) ** 2) / 2, dim=1, keepdim=True)
            for loc, scale in zip(self.locs, self.scales)
        ]
        return -torch.logsumexp(-torch.stack(energies, dim=1), dim=1).squeeze(-1)


@pytest.mark.parametrize(
    "name, fsampler, num_steps",
    [
        ("langevin", lambda: src.mcmc.langevin.LangevinSampler(lr=0.1), 100),
        ("mala", lambda: src.mcmc.mala.MALASampler(lr=0.1), 100),
        (
            "tempered langevin",
            lambda: src.mcmc.tempered_transitions.TemperedTransitions(
                mc_dynamics=src.mcmc.langevin.LangevinSampler(lr=0.5)
            ),
            10,
        ),
        (
            "tempered mala",
            lambda: src.mcmc.tempered_transitions.TemperedTransitions(
                mc_dynamics=src.mcmc.mala.MALASampler(lr=0.5)
            ),
            10,
        ),
    ],
)
def test_normal_stats(
    name, fsampler, num_steps, mean_tolerance=0.05, std_tolerance=0.05, num_samples=5000
):
    """Tests that samplers derive correct normal stats."""
    net = NormalNet()
    sampler = fsampler()
    samples = []
    x = None
    for _ in range(num_steps):
        x = net.sample_fantasy(
            x, num_mc_steps=1, num_samples=num_samples, mc_dynamics=sampler
        )
        samples.append(x)
    all_x = torch.stack(samples[num_steps // 2 :])
    mean = torch.mean(all_x)
    std = torch.std(all_x, unbiased=True)
    assert abs(mean) < mean_tolerance, f"Mean via {name} was not accurate"
    expected = np.sqrt(net.variance())
    assert (
        abs(std - expected) < std_tolerance
    ), f"Standard Deviation ({std}) via {name} was not accurate ({expected})"


test_samplers = {
    "langevin": lambda: src.mcmc.langevin.LangevinSampler(lr=0.5),
    "mala": lambda: src.mcmc.mala.MALASampler(lr=0.5),
    "tempered langevin": lambda: src.mcmc.tempered_transitions.TemperedTransitions(
        mc_dynamics=src.mcmc.langevin.LangevinSampler(lr=0.5)
    ),
    "tempered mala": lambda: src.mcmc.tempered_transitions.TemperedTransitions(
        mc_dynamics=src.mcmc.mala.MALASampler(lr=0.5)
    ),
}


def plot_samplers(fsamplers=test_samplers, num_steps=100, net_factory=MixtureNormalNet):
    """Plots samplers to compare them visually."""
    f, ax = plt.subplots(1, 2)
    net = net_factory()
    vals = {}
    for name, fsampler in fsamplers.items():
        samples = []
        sampler = fsampler()
        x = None
        for _ in range(num_steps):
            x = net.sample_fantasy(
                x, num_mc_steps=1, num_samples=1, mc_dynamics=sampler
            )
            samples.append(x)
        all_x = torch.stack(samples[num_steps // 2 :])
        mean = torch.mean(all_x)
        var = torch.var(all_x)
        print(f"{name}:\n" f" - mean: {mean}\n" f" - var: {var}")
        x_np = all_x.detach().numpy().flatten()
        vals[name] = x_np
        ax[0].scatter(np.arange(x_np.size), x_np, label=name, alpha=0.75)
    labels = list(vals.keys())
    xs = np.stack(vals.values(), axis=-1)
    ax[1].hist(xs, label=labels, density=True)
    num_xs = 100
    x_range = np.linspace(xs.min(), xs.max(), num_xs)
    y_range = torch.exp(-net(torch.tensor(x_range.reshape([-1, 1]))))
    cum_y = torch.sum(y_range)
    y_range *= num_xs / (cum_y * (xs.max() - xs.min()))
    ax[1].plot(x_range, y_range)
    f.legend()
    plt.show()


def test_serialization(fsamplers=test_samplers):
    """Tests serialization and deserialization of samplers"""
    for name, fsampler in fsamplers.items():
        sampler = fsampler()
        state = sampler.state_dict()
        sampler_new = fsampler()
        sampler_new.load_state_dict(state)
        assert (
            state == sampler_new.state_dict()
        ), f"{name} state did not seem to serialize correctly"
