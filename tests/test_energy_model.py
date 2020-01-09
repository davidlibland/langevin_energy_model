import numpy as np
import torch

from src.model import SimpleEnergyModel


def visual_test_sample_fantasy_(
    fig,
    num_layers=2,
    num_units=2,
    num_steps=10,
    num_mc_steps=100,
    num_chains=10,
    lr=1e-3,
    sigma=1,
    net=None,
):
    fig.clear()
    ax = fig.subplots(1, 1)
    if net is None:
        net = SimpleEnergyModel(
            num_inputs=1, num_layers=num_layers, num_units=num_units
        )
    X = torch.randn(num_chains, 1, dtype=torch.float)
    X_s = []
    for _ in range(num_steps):
        X = net.sample_fantasy(X, num_mc_steps=num_mc_steps, lr=lr, sigma=sigma)
        X_s.append(X)
    X = torch.stack(X_s, dim=0)
    X = X.detach().numpy()
    ax.hist(X.reshape([-1]), density=True, label="Data")
    x_min = X.min()
    x_max = X.max()
    xs = np.linspace(x_min, x_max, 100)
    ys_ = torch.exp(-net(torch.as_tensor(xs.reshape([-1, 1]), dtype=torch.float)))
    Z = ys_.mean().detach().numpy() * (x_max - x_min)
    ys = ys_.detach().numpy().reshape(-1) / Z
    ax.plot(xs, ys, label="Actual")
    ax.legend()
    return net
