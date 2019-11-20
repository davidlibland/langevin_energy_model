import numpy as np
from ray.tune.schedulers import ASHAScheduler
from ray import tune

from distributions.core import Distribution, Normal
from distributions.mnist import get_approx_mnist_distribution
from distributions.digits import get_approx_digit_distribution
from distributions.small_digits import get_approx_sm_digit_distribution
from distributions.small_patterns import get_approx_pattern_distribution
from hparam_sweep import get_energy_trainer
from model import ConvEnergyModel
from model import SimpleEnergyModel, ResnetEnergyModel


def setup_1d(**kwargs):
    dist = Distribution.mixture([
        Normal(np.array([-3])),
        Normal(np.array([3])),
        Normal(np.array([15])),
    ])
    net = SimpleEnergyModel(1, 3, 25)
    return dist, net


def setup_2d(**kwargs):
    dist = Distribution.mixture([
        Normal(np.array([-3, 2])),
        Normal(np.array([3, 5])),
        Normal(np.array([15, 1])),
    ])
    net = SimpleEnergyModel(2, 3, 25)
    return dist, net


def setup_patterns(patterns=("checkerboard_2x2", "diagonal_gradient_2x2"), **kwargs):
    dist = get_approx_pattern_distribution(patterns)
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    net = ResnetEnergyModel((1, 2, 2), 2, 3, 12, prior_scale=5*scale)
    return dist, net


def setup_sm_digits_simple(**kwargs):
    dist = get_approx_sm_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    net = SimpleEnergyModel(16, 3, 60, prior_scale=5*scale)
    return dist, net


def setup_sm_digits(model="conv", n_hidden=12, **kwargs):
    dist = get_approx_sm_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    if model == "resnet":
        net = ResnetEnergyModel((1, 4, 4), 2, 2, n_hidden, prior_scale=5*scale)
    else:
        net = ConvEnergyModel((1, 4, 4), 2, n_hidden, prior_scale=5*scale)
    return dist, net


def setup_digits(model="conv", n_hidden=25, **kwargs):
    dist = get_approx_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    if model == "resnet":
        net = ResnetEnergyModel((1, 8, 8), 3, 2, n_hidden, prior_scale=5*scale)
    else:
        net = ConvEnergyModel((1, 8, 8), 3, n_hidden, prior_scale=5*scale)
    return dist, net


def setup_mnist(n_hidden=25, **kwargs):
    dist = get_approx_mnist_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    net = ResnetEnergyModel((1, 28, 28), 3, 2, n_hidden, prior_scale=scale)
    return dist, net


if __name__ == '__main__':
    analysis = tune.run(
        get_energy_trainer(setup_sm_digits),
        config={
            "lr": tune.loguniform(1e-4, 1e-1),
            "num_epochs": tune.randint(10, 20),
            "num_mc_steps": tune.randint(1, 20),
            "n_hidden": tune.randint(4, 48),
            "model": tune.choice(["conv", "resnet"])
        },
        scheduler=ASHAScheduler(metric="loss_ais", mode="min"),
        num_samples=2,
        checkpoint_freq=1,
        checkpoint_at_end=True,
    )
    print(analysis.get_best_config(metric="loss_ais", mode="min"))
