import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

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
from utils.logging import tb_writer


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


def setup_sm_digits(model="conv"):
    dist = get_approx_sm_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    if model == "resnet":
        net = ResnetEnergyModel((1, 4, 4), 2, 2, 12, prior_scale=5*scale)
    else:
        net = ConvEnergyModel((1, 4, 4), 2, 12, prior_scale=5*scale)
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


def main(lr=1e-2, num_epochs=10, fname="samples", show=True,
         num_mc_steps=10, num_sample_mc_steps=10000, sample_beta=0.1):
    # dist, net = setup_digits("resnet")
    dist, net = setup_sm_digits_simple()
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
    tb_writer.add_figure(tag="data", figure=fig)
    plt.close(fig)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-1)

    sampler = MALASampler(lr=0.1)
    samples = None

    def save_images(model_sample, global_step, epoch, validation, net, **kwargs):
        nonlocal samples
        if validation and epoch % 5 != 0:
            fig = plt.figure()
            net.eval()
            dist.visualize(fig, model_sample.cpu(), energy)
            if show:
                plt.show()
            if fname:
                fig.savefig(os.path.join(RUN_DIR, f"{fname}_{epoch}.png"))
            tb_writer.add_figure(tag="negative_samples", figure=fig, global_step=global_step)
            plt.close(fig)
        elif validation:
            fig = plt.figure()
            net.eval()
            samples = net.sample_fantasy(x=samples,
                                         num_mc_steps=num_sample_mc_steps,
                                         beta=sample_beta,
                                         mc_dynamics=sampler,
                                         num_samples=36
                                         ).detach().cpu()
            dist.visualize(fig, samples, energy)
            if show:
                plt.show()
            if fname:
                fig.savefig(os.path.join(RUN_DIR, f"{fname}_{epoch}.png"))
            tb_writer.add_figure(tag="model_samples", figure=fig, global_step=global_step)
            plt.close(fig)

    def save_model(net, epoch, validation, **kwargs):
        if validation and epoch % 5 != 0:
            ckpt_fn = os.path.join(RUN_DIR, f"ckpt_{epoch}.pkl")
            torch.save({
                    "epoch": epoch,
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, ckpt_fn
            )

    ais_loss = AISLoss(tb_writer=tb_writer, log_z_update_interval=9,
                       particle_filter=False, optimizer=optimizer,
                       lr_decay_factor=0.95)

    net, optimizer = train(net, dataset, num_epochs=num_epochs,
                           optimizer=optimizer,
                           num_mc_steps=num_mc_steps,
                           ckpt_callbacks=[save_images, ais_loss, save_model])

if __name__ == '__main__':
    main(lr=1e-2, num_epochs=1000, show=False, num_mc_steps=10)
