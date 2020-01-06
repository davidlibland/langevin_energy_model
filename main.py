import numpy as np
from ray.tune.schedulers import ASHAScheduler
from ray import tune

from distributions.core import Distribution, Normal
from distributions.mnist import get_mnist_distribution
from distributions.digits import get_digit_distribution
from distributions.small_digits import get_sm_digit_distribution
from distributions.small_patterns import get_pattern_distribution
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
    dist = get_pattern_distribution(patterns)
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    net = ResnetEnergyModel((1, 2, 2), 2, 3, 12, prior_scale=5*scale)
    return dist, net


def setup_sm_digits_simple(**kwargs):
    dist = get_sm_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    net = SimpleEnergyModel(16, 3, 60, prior_scale=5*scale)
    return dist, net


def setup_sm_digits(model="conv", n_hidden=12, prior_scale_factor=5, **kwargs):
    dist = get_sm_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    if model == "resnet":
        net = ResnetEnergyModel((1, 4, 4), 2, 2, n_hidden, prior_scale=prior_scale_factor*scale)
    else:
        net = ConvEnergyModel((1, 4, 4), 2, n_hidden, prior_scale=prior_scale_factor*scale)
    return dist, net


def setup_digits(model="conv", n_hidden=25, prior_scale_factor=5, **kwargs):
    dist = get_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    if model == "resnet":
        net = ResnetEnergyModel((1, 8, 8), 3, 2, n_hidden, prior_scale=prior_scale_factor*scale)
    else:
        net = ConvEnergyModel((1, 8, 8), 3, n_hidden, prior_scale=prior_scale_factor*scale)
    return dist, net


def setup_mnist(n_hidden=25, prior_scale_factor=5, **kwargs):
    dist = get_mnist_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    net = ResnetEnergyModel((1, 28, 28), 3, 2, n_hidden, prior_scale=prior_scale_factor*scale)
    return dist, net


if __name__ == '__main__':
    analysis = tune.run(
        get_energy_trainer(setup_sm_digits),
        config={
            "lr": tune.loguniform(1e-5, 1e-1),
            "n_hidden": tune.randint(4, 512),
            "model": tune.choice(["conv", "resnet"]),
            "batch_size": tune.randint(128, 1024),
            "prior_scale_factor": tune.loguniform(1e-1, 1e2)
        },
        scheduler=ASHAScheduler(metric="loss_ais", mode="min"),
        num_samples=2,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        resources_per_trial={
             "gpu": 1,
         }
    )
    df = analysis.dataframe()
    save_filename = input("Where to save the csv?")
    df.to_csv(save_filename)
    print(analysis.get_best_config(metric="loss_ais", mode="min"))
