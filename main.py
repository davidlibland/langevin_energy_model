import numpy as np
from ray.tune.schedulers import ASHAScheduler
from ray import tune

from src.distributions.core import Distribution, Normal
from src.distributions.mnist import get_mnist_distribution
from src.distributions.digits import get_digit_distribution
from src.distributions.small_digits import get_sm_digit_distribution
from src.distributions.small_patterns import get_pattern_distribution
from src.hparam_sweep import get_energy_trainer
from src.model import ConvEnergyModel
from src.model import SimpleEnergyModel, ResnetEnergyModel


def setup_1d(**kwargs):
    dist = Distribution.mixture(
        [Normal(np.array([-3])), Normal(np.array([3])), Normal(np.array([15]))]
    )
    net = SimpleEnergyModel(1, 3, 25)
    return dist, net


def setup_2d(**kwargs):
    dist = Distribution.mixture(
        [Normal(np.array([-3, 2])), Normal(np.array([3, 5])), Normal(np.array([15, 1]))]
    )
    net = SimpleEnergyModel(2, 3, 25)
    return dist, net


def setup_patterns(patterns=("checkerboard_2x2", "diagonal_gradient_2x2"), **kwargs):
    dist = get_pattern_distribution(patterns)
    n = 1000
    scale = np.sqrt((dist.rvs(n) ** 2).sum() / n)
    net = ResnetEnergyModel((1, 2, 2), 2, 3, 12, prior_scale=5 * scale)
    return dist, net


def setup_sm_digits_simple(**kwargs):
    dist = get_sm_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n) ** 2).sum() / n)
    net = SimpleEnergyModel(16, 3, 60, prior_scale=5 * scale)
    return dist, net


def setup_sm_digits(model="conv", n_hidden=12, prior_scale_factor=5, **kwargs):
    dist = get_sm_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n) ** 2).sum() / n)
    scale = 1
    if model == "resnet":
        net = ResnetEnergyModel(
            (1, 4, 4), 2, 2, n_hidden, prior_scale=prior_scale_factor * scale
        )
    else:
        net = ConvEnergyModel(
            (1, 4, 4), 2, n_hidden, prior_scale=prior_scale_factor * scale
        )
    return dist, net


def setup_digits(model="conv", n_hidden=25, prior_scale_factor=5, **kwargs):
    dist = get_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n) ** 2).sum() / n)
    if model == "resnet":
        net = ResnetEnergyModel(
            (1, 8, 8), 3, 2, n_hidden, prior_scale=prior_scale_factor * scale
        )
    else:
        net = ConvEnergyModel(
            (1, 8, 8), 3, n_hidden, prior_scale=prior_scale_factor * scale
        )
    return dist, net


def setup_mnist(n_hidden=25, prior_scale_factor=5, **kwargs):
    dist = get_mnist_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n) ** 2).sum() / n)
    net = ResnetEnergyModel(
        (1, 28, 28), 3, 2, n_hidden, prior_scale=prior_scale_factor * scale
    )
    return dist, net


def stop_on_low_ais_ess(trial_id, result):
    """Stops trials if the effective sample size falls too low."""
    return result["ais_effective_sample_size"] < .1


if __name__ == "__main__":
    analysis = tune.run(
        get_energy_trainer(setup_sm_digits),
        name="sm_digit_non_convergent",
        config={
            "lr": 1e-3, #1e-2, #tune.loguniform(1e-3, 3e-2),
            "weight_decay": 0, #1e-3, #6, #tune.loguniform(1e-1, 1e1),
            "n_hidden": 64, #tune.randint(32, 48),
            "model": tune.choice(["conv"]),#, "resnet"]),
            "batch_size": 1000, #tune.randint(256, 1024),
            "prior_scale_factor": 1, #5, #1,
            "sampler": tune.choice(["langevin"]), #, "langevin", "tempered langevin", "tempered mala"]),
            "sampler_beta_schedule_start": .6, # tune.loguniform(1e-1, 0.6),
            "sampler_lr": 1, #tune.loguniform(3e-2, 15), #2
            "num_mc_steps": 100,
            "num_sample_mc_steps": 100
        },
        scheduler=ASHAScheduler(metric="loss_ais", mode="min", max_t=10**4),
        num_samples=1,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        # stop=stop_on_low_ais_ess,
        # resources_per_trial={"gpu": 1},
        # resume="PROMPT"
        # restore="/Users/dlibland/ray_results/sm_digit_fives_/EnergyTrainer_0_model=resnet,sampler=mala_2020-02-24_09-39-38fnpaf60f/checkpoint_104"
    )
    df = analysis.dataframe()
    save_filename = input("Where to save the csv?")
    df.to_csv(save_filename)
    print(analysis.get_best_config(metric="loss_ais", mode="min"))
