"""
This file adds some MC tests for samplers.
"""
import pytest
import torch

import mcmc.langevin
import mcmc.mala
import model


class NormalNet(model.BaseEnergyModel):
    def __init__(self, scale=1, mc_dynamics=None):
        num_features=1
        super().__init__(num_features=num_features, mc_dynamics=mc_dynamics)
        self.scale = scale

    def energy(self, x):
        energy = torch.sum(((x/self.scale)**2)/2, dim=1, keepdim=True)
        return energy

    def variance(self, beta=1):
        """Computes the variance at a given beta."""
        inv_var = beta/self.scale**2 + 1/self.prior_scale**2
        return 2/inv_var


@pytest.mark.parametrize("sampler", [mcmc.langevin.LangevinSampler, mcmc.mala.MALASampler])
def test_normal_stats(sampler, mean_tolerance=0.005, var_tolerance=0.05, num_steps=1000, num_samples=1000):
    """Tests that samplers derive correct normal stats."""
    net = NormalNet(mc_dynamics=sampler(lr=0.1))
    samples = []
    for _ in range(num_steps):
        x = net.sample_fantasy(num_mc_steps=1, num_samples=num_samples)
        samples.append(x)
    all_x = torch.stack(samples[num_steps//2:])
    mean = torch.mean(all_x)
    var = torch.var(all_x)
    assert abs(mean) < mean_tolerance, "Mean was not accurate"
    assert abs(var - net.variance()) < var_tolerance, "Variance was not accurate"
