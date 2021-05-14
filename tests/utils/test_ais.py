import numpy as np
import scipy.special as scisp
import torch
import torch.nn as nn

import energy_model.utils.beta_schedules
from energy_model.mcmc.langevin import LangevinSampler
from energy_model.mcmc.mala import MALASampler
from energy_model.model import BaseEnergyModel
from energy_model.model import LANG_INIT_NS
from energy_model.utils.ais import AISLoss


class DiagonalNormalModel(BaseEnergyModel):
    def __init__(self, num_features, prior_scale=LANG_INIT_NS):
        super().__init__(input_shape=(num_features,), prior_scale=prior_scale)
        self.mean = nn.Parameter(torch.randn(1, num_features))
        self.log_scale = nn.Parameter(torch.randn(1, num_features))

    def energy(self, x):
        log_z = (0.5 * np.log(2 * np.pi) + self.log_scale).sum()
        result = (0.5 * ((x - self.mean) / torch.exp(self.log_scale)) ** 2).sum(
            dim=1, keepdim=True
        ) + log_z
        return result.squeeze(-1)

    def loss(self, x):
        denom = torch.exp(2 * self.log_scale) + self.prior_scale ** 2
        full_mean = self.mean * self.prior_scale ** 2 / denom
        log_full_scale = (
            self.log_scale + np.log(self.prior_scale) - 0.5 * torch.log(denom)
        )
        log_z = (0.5 * np.log(2 * np.pi) + log_full_scale).sum()
        return (
            0.5 * (((x - full_mean) / torch.exp(log_full_scale)).sum(dim=1)) ** 2
            + log_z
        ).mean()


class MockLogger:
    def __init__(self):
        self.logs = dict()

    def __call__(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.logs:
                self.logs[k] = []
            self.logs[k].append(v)


def test_ais_loss(num_features=2, num_samples=200, num_chains=1000, lr=0.1):
    net = DiagonalNormalModel(num_features)
    X = torch.randn(num_samples, num_features)
    N = num_chains
    Y = torch.randn(N, num_features)
    Y_p = (
        N
        * torch.exp(-(Y ** 2).sum(dim=1, keepdim=True) / 2)
        / (np.sqrt(2 * np.pi) ** num_features)
    )
    exact_loss = net.loss(X)
    mc_loss = net(X).mean() + (scisp.logsumexp(-net(Y).detach(), b=1 / Y_p))
    print(exact_loss)
    print(mc_loss)

    logger = MockLogger()
    ais_loss_obj = AISLoss(logger=logger, num_chains=N, mc_dynamics=MALASampler(lr=lr),)

    ais_loss_obj(net=net, global_step=ais_loss_obj.log_z_update_interval, data_sample=X)
    print(logger.logs["loss_ais"])

    ais_loss_obj = AISLoss(
        logger=logger, num_chains=N, mc_dynamics=LangevinSampler(lr=lr),
    )

    ais_loss_obj(net=net, global_step=ais_loss_obj.log_z_update_interval, data_sample=X)
    print(logger.logs["loss_ais"])
