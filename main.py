import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

from distributions.core import Distribution, Normal
from distributions.mnist import get_approx_mnist_distribution
from distributions.digits import get_approx_digit_distribution
from model import BaseEnergyModel, SimpleEnergyModel, ConvEnergyModel, ResnetEnergyModel


def train(net: BaseEnergyModel, dataset: data.Dataset, num_steps=10, lr=1e-2,
          batch_size=100, model_sample=None, optimizer=None, num_mc_steps=20,
          mc_lr=1, verbose=True):
    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True
    )
    # Determine the device type:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(f"training on {len(dataset)} samples using {device}.")
    for step in range(num_steps):
        losses = []
        for i_batch, sample_batched in enumerate(dataloader):
            data_sample = sample_batched[0].to(device)
            if model_sample is None:
                model_sample = data_sample

            net.eval()
            for mc_step in range(num_mc_steps):
                model_sample = net.sample_fantasy(model_sample, lr=mc_lr).detach()

            # Forward gradient:
            net.train()
            net.zero_grad()
            loss = (net(data_sample) - net(model_sample)).mean()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_value_(net.parameters(), 1e2)
            optimizer.step()
            losses.append(loss)
            if verbose:
                print(f"on epoch {step}, batch {i_batch}, loss: {loss}")
        print(f"on epoch {step}, loss: {sum(losses)/len(losses)}")
    return model_sample


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
    net = ResnetEnergyModel((1, 8, 8), 3, 2, 25)
    return dist, net


def setup_mnist():
    dist = get_approx_mnist_distribution()
    net = ResnetEnergyModel((1, 28, 28), 3, 2, 25)
    return dist, net


def main(lr=1e-2, num_steps=10, fname="samples", show=True, num_saves=100,
         num_mc_steps=10):
    dist, net = setup_digits()
    samples = dist.rvs(1000)
    print(samples.shape)
    dataset = data.TensorDataset(torch.tensor(samples, dtype=torch.float))
    energy = lambda x: net(torch.tensor(x, dtype=torch.float)).detach().numpy()
    fig = plt.figure()
    dist.visualize(fig, samples, energy)
    if show:
        plt.show()
    if fname:
        fig.savefig(f"{fname}_data.png")
    plt.close(fig)

    model_sample = None
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    i = 1
    while i % num_saves:
        model_sample = train(net, dataset, num_steps=num_steps,
                             model_sample=model_sample, optimizer=optimizer,
                             num_mc_steps=num_mc_steps)
        fig = plt.figure()
        dist.visualize(fig, model_sample.cpu(), energy)
        if show:
            plt.show()
        if fname:
            fig.savefig(f"{fname}_{i}.png")
        plt.close(fig)
        i += 1
        if i % num_saves == 0:
            if input("Train more?") != "y":
                break
            i += 1


if __name__ == '__main__':
    main(lr=1e-3, num_steps=1, num_saves=100, show=False, num_mc_steps=5)
