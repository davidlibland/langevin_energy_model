import os

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

from distributions.core import Distribution, Normal
from distributions.mnist import get_approx_mnist_distribution
from distributions.digits import get_approx_digit_distribution
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


def setup_digits():
    dist = get_approx_digit_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    net = ResnetEnergyModel((1, 8, 8), 3, 2, 25, prior_scale=5*scale)
    return dist, net


def setup_mnist():
    dist = get_approx_mnist_distribution()
    n = 1000
    scale = np.sqrt((dist.rvs(n)**2).sum()/n)
    net = ResnetEnergyModel((1, 28, 28), 3, 2, 25, prior_scale=scale)
    return dist, net


def main(lr=1e-2, num_epochs=10, fname="samples", show=True,
         num_mc_steps=10):
    dist, net = setup_2d()
    samples = dist.rvs(1000)
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

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)

    def save_images(model_sample, global_step, epoch, validation, **kwargs):
        if validation:
            fig = plt.figure()
            dist.visualize(fig, model_sample.cpu(), energy)
            if show:
                plt.show()
            if fname:
                fig.savefig(os.path.join(RUN_DIR, f"{fname}_{epoch}.png"))
            tb_writer.add_figure(tag="data", figure=fig, global_step=global_step)
            plt.close(fig)

    ais_loss = AISLoss(tb_writer=tb_writer, log_z_update_interval=10, lr=.1)

    net, optimizer = train(net, dataset, num_epochs=num_epochs,
                           optimizer=optimizer,
                           num_mc_steps=num_mc_steps,
                           ckpt_callbacks=[save_images, ais_loss])

if __name__ == '__main__':
    main(lr=1e-3, num_epochs=1000, show=False, num_mc_steps=100)
