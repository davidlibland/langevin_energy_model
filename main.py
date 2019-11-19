import csv
import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from ray.tune.schedulers import ASHAScheduler

from distributions.core import Distribution, Normal
from distributions.mnist import get_approx_mnist_distribution
from distributions.digits import get_approx_digit_distribution
from distributions.small_digits import get_approx_sm_digit_distribution
from distributions.small_patterns import get_approx_pattern_distribution
from mcmc.mala import MALASampler
from model import ConvEnergyModel
from model import SimpleEnergyModel, ResnetEnergyModel
from training_loop import train
from utils.ais import AISLoss
from utils.logging import RUN_DIR
from ray import tune


def setup_1d():
    dist = Distribution.mixture([
        Normal(np.array([-3])),
        Normal(np.array([3])),
        Normal(np.array([15])),
    ])
    net = SimpleEnergyModel(1, 3, 25)
    return dist, net


def setup_2d():
    dist = Distribution.mixture([
        Normal(np.array([-3, 2])),
        Normal(np.array([3, 5])),
        Normal(np.array([15, 1])),
    ])
    net = SimpleEnergyModel(2, 3, 25)
    return dist, net


def setup_patterns(patterns=("checkerboard_2x2", "diagonal_gradient_2x2")):
    dist = get_approx_pattern_distribution(patterns)
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    net = ResnetEnergyModel((1, 2, 2), 2, 3, 12, prior_scale=5*scale)
    return dist, net


def setup_sm_digits_simple():
    dist = get_approx_sm_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    net = SimpleEnergyModel(16, 3, 60, prior_scale=5*scale)
    return dist, net


def setup_sm_digits(model="conv", n_hidden=12):
    dist = get_approx_sm_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    if model == "resnet":
        net = ResnetEnergyModel((1, 4, 4), 2, 2, n_hidden, prior_scale=5*scale)
    else:
        net = ConvEnergyModel((1, 4, 4), 2, n_hidden, prior_scale=5*scale)
    return dist, net


def setup_digits(model="conv"):
    dist = get_approx_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    if model == "resnet":
        net = ResnetEnergyModel((1, 8, 8), 3, 2, 25, prior_scale=5*scale)
    else:
        net = ConvEnergyModel((1, 8, 8), 3, 25, prior_scale=5*scale)
    return dist, net


def setup_mnist():
    dist = get_approx_mnist_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    net = ResnetEnergyModel((1, 28, 28), 3, 2, 25, prior_scale=scale)
    return dist, net


def training_entrypoint(config):
    lr = config.get("lr", 1e-2)
    num_epochs = config.get("num_epochs", 10)
    num_mc_steps = config.get("num_mc_steps", 10)
    num_sample_mc_steps = config.get("num_sample_mc_steps", 1000)
    sample_beta = config.get("sample_beta", 0.1)
    n_hidden = config.get("n_hidden", 12)
    model = config.get("model", "conv")
    fname = "samples"
    show = False
    os.makedirs(RUN_DIR, exist_ok=True)

    # dist, net = setup_digits("resnet")
    dist, net = setup_sm_digits(model=model, n_hidden=n_hidden)
    samples = dist.rvs(10000)
    print(samples.shape)
    dataset = data.TensorDataset(torch.tensor(samples, dtype=torch.float))
    energy = lambda x: net(torch.tensor(x, dtype=torch.float)).detach().numpy()
    fig = plt.figure()
    dist.visualize(fig, samples, energy)
    if show:
        plt.show()
    if fname:
        fig.savefig(os.path.join(RUN_DIR, f"{fname}_data.png"))
    plt.close(fig)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-1)

    sampler = MALASampler(lr=0.1)

    logdir = tune.track.get_session().logdir
    def save_images(net):
        fig = plt.figure()
        net.eval()
        samples = net.sample_fantasy(x=None,
                                     num_mc_steps=num_sample_mc_steps,
                                     beta=sample_beta,
                                     mc_dynamics=sampler,
                                     num_samples=36
                                     ).detach().cpu()
        dist.visualize(fig, samples, energy)
        if show:
            plt.show()
        if fname:
            fig.savefig(os.path.join(RUN_DIR, f"samples.png"))
        plt.close(fig)

    def save_model(net):
        ckpt_fn = os.path.join(RUN_DIR, f"model.pkl")
        torch.save({
                "epoch": num_epochs,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict()
            }, ckpt_fn
        )

    ais_loss = AISLoss(logger=tune.track.log, log_z_update_interval=9)

    net, optimizer = train(net, dataset, num_epochs=num_epochs,
                           optimizer=optimizer,
                           num_mc_steps=num_mc_steps,
                           step_callbacks=[ais_loss])
    save_model(net=net)
    save_images(net=net)


if __name__ == '__main__':
    analysis = tune.run(
        training_entrypoint,
        config={
            "lr": tune.loguniform(1e-4, 1e-1),
            "num_epochs": tune.randint(10, 500),
            "num_mc_steps": tune.randint(1, 20),
            "n_hidden": tune.randint(4, 128),
            "model": tune.choice(["conv", "resnet"])
        },
        scheduler=ASHAScheduler(metric="loss_ais", mode="min"),
        num_samples=20
    )
    print(analysis.get_best_config(metric="loss_ais", mode="min"))
