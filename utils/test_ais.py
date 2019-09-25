import numpy as np
import torch
import torch.nn as nn
import scipy.special as scisp

from model import BaseEnergyModel
from model import LANG_INIT_NS
from utils.ais import AISLoss


class DiagonalNormalModel(BaseEnergyModel):
    def __init__(self, num_features, prior_scale=LANG_INIT_NS):
        super().__init__(num_features=num_features, prior_scale=prior_scale)
        self.mean = nn.Parameter(torch.randn(1, num_features))
        self.log_scale = nn.Parameter(torch.randn(1, num_features))

    def energy(self, x):
        log_z = (0.5*np.log(2*np.pi)+self.log_scale).sum()
        result = (0.5*((x - self.mean)/torch.exp(self.log_scale))**2).sum(dim=1, keepdim=True)+log_z
        return result

    def loss(self, x):
        denom = torch.exp(2*self.log_scale)+self.prior_scale**2
        full_mean = self.mean*self.prior_scale**2/denom
        log_full_scale = self.log_scale + np.log(self.prior_scale) - 0.5*torch.log(denom)
        log_z = (0.5*np.log(2*np.pi)+log_full_scale).sum()
        return (0.5*(((x - full_mean)/torch.exp(log_full_scale)).sum(dim=1))**2+log_z).mean()


class MockTBWriter:
    def __init__(self):
        self.loss = dict()

    def add_scalar(self, name, scalar_value, global_step):
        self.loss[(name, global_step)] = scalar_value

    def add_histogram(self, *args, **kwargs):
        pass


def test_ais_loss(num_features=2, num_samples=200, num_chains=1000):
    net = DiagonalNormalModel(num_features=num_features)
    X = torch.randn(num_samples, num_features)
    N = num_chains
    Y = torch.randn(N, num_features)
    Y_p = N*torch.exp(-(Y**2).sum(dim=1, keepdim=True)/2)/(np.sqrt(2*np.pi)**num_features)
    exact_loss = net.loss(X)
    mc_loss = net(X).mean() + (scisp.logsumexp(-net(Y).detach(), b=1/Y_p))
    print(exact_loss)
    print(mc_loss)

    tb_writer = MockTBWriter()
    beta_schedule = AISLoss.build_schedule(
        ("arith", .01, 200),
        ("geom", 1., 1000)
    )
    ais_loss_obj = AISLoss(tb_writer=tb_writer, beta_schedule=beta_schedule, num_chains=N)

    ais_loss_obj(net=net, global_step=ais_loss_obj.log_z_update_interval, data_sample=X)
    print(tb_writer.loss)
