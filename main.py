import itertools
import os

import ConfigSpace as CS
import ray.tune.schedulers
import ray.tune.suggest
from ray import tune
import numpy as np

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
    net = ResnetEnergyModel((1, 2, 2), 2, 3, 12)
    return dist, net


def setup_sm_digits_simple(**kwargs):
    dist = get_sm_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n) ** 2).sum() / n)
    net = SimpleEnergyModel(16, 3, 60)
    return dist, net


def setup_sm_digits(model="conv", n_hidden=12, prior_scale_factor=5, **kwargs):
    dist = get_sm_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n) ** 2).sum() / n)
    if model == "resnet":
        net = ResnetEnergyModel((1, 4, 4), 2, 2, n_hidden)
    else:
        net = ConvEnergyModel((1, 4, 4), 2, n_hidden)
    print(f"net: {net.input_shape}")
    return dist, net


def setup_digits(model="conv", n_hidden=25, prior_scale_factor=5, **kwargs):
    dist = get_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n) ** 2).sum() / n)
    if model == "resnet":
        net = ResnetEnergyModel((1, 8, 8), 3, 2, n_hidden)
    else:
        net = ConvEnergyModel((1, 8, 8), 3, n_hidden)
    return dist, net


def setup_mnist(n_hidden=25, prior_scale_factor=5, **kwargs):
    dist = get_mnist_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n) ** 2).sum() / n)
    net = ResnetEnergyModel((1, 28, 28), 3, 2, n_hidden)
    return dist, net


def stop_on_low_ais_ess(trial_id, result):
    """Stops trials if the effective sample size falls too low."""
    return result["ais_effective_sample_size"] < 0.1


def stop_on_low_data_erf(trial_id, result):
    """Stops trials if the data erf falls too low."""
    return result["data_erf"] < -1


def should_stop(*criteria):
    """Helper function to compound multiple criteria"""

    def helper(trial_id, result):
        return any(criterion(trial_id, result) for criterion in criteria)

    return helper


if __name__ == "__main__":
    # Create hyper-parameter optimization client:
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("lr", lower=1e-6, upper=1e-3, log=True)
    )
    # Unused for mnist:
    # config_space.add_hyperparameter(
    #     CS.UniformFloatHyperparameter("prior_scale_factor", lower=1, upper=3, log=True)
    # )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("sampler_lr", lower=1e-5, upper=1e-1, log=True)
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter("num_mc_steps", lower=200, upper=1000, log=True)
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "sampler_beta_target", lower=1e5, upper=1e10, log=True
        )
    )

    # The following are only used for tempered transitions:

    # config_space.add_hyperparameter(
    #     CS.UniformFloatHyperparameter(
    #         "sampler_beta_min", lower=1e-1, upper=0.6, log=True
    #     )
    # )
    # config_space.add_hyperparameter(
    #     CS.UniformIntegerHyperparameter("num_tempered_transitions", lower=1, upper=10)
    # )

    experiment_metrics = dict(metric="loss", mode="min")
    bohb_hyperband = ray.tune.schedulers.HyperBandForBOHB(
        time_attr="training_iteration", max_t=1000, **experiment_metrics
    )
    bohb_search = ray.tune.suggest.TuneBOHB(
        config_space, max_concurrent=4, **experiment_metrics
    )

    # Run the experiments:
    experiment_name = "mnist_bohb"
    analysis = tune.run(
        get_energy_trainer(setup_mnist),
        name=experiment_name,
        config={
            "ais_update_interval": 18,
            "ais_max_interpolants": 5000,
            "ais_num_interpolants": 500,
            "weight_decay": 0,
            "n_hidden": 64,
            "model": ["conv", "resnet"],
            "batch_size": 1024,
            "sampler": "langevin",
            "num_sample_mc_steps": 1500,
        },
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        # scheduler=ray.tune.schedulers.ASHAScheduler(metric="loss_ais", mode="min", max_t=10 ** 4),
        num_samples=50,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        stop=should_stop(stop_on_low_ais_ess, stop_on_low_data_erf),
        # reuse_actors=True,
        resources_per_trial={"gpu": 1},
        # resume="PROMPT"
        # restore="/Users/dlibland/ray_results/sm_digit_fives_/EnergyTrainer_0_model=resnet,sampler=mala_2020-02-24_09-39-38fnpaf60f/checkpoint_104"
    )

    # Save the tune results:
    df = analysis.dataframe()
    experiment_dir = analysis._experiment_dir
    results_filename = os.path.join(experiment_dir, "results.csv")
    df.to_csv(results_filename)
    print(analysis.get_best_config(metric="loss_ais", mode="min"))
