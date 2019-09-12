import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

from distributions.core import Distribution, Normal
from distributions.mnist import get_approx_mnist_distribution
from model import SimpleEnergyModel, ConvEnergyModel


def train(net: SimpleEnergyModel, dataset: data.Dataset, num_steps=10, lr=1e-2,
          batch_size=100, model_sample=None,
          verbose=True):
    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-3)
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

            model_sample = net.sample_fantasy(model_sample, lr=1e-1).detach()

            # Forward gradient:
            net.zero_grad()
            loss = (net(data_sample) - net(model_sample)).mean()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_value_(net.parameters(), 1e2)
            optimizer.step()
            losses.append(loss)
            if verbose:
                print(f"on epoch {step}, batch {i_batch}, loss: {loss}")
        print(f"on epoch {step}, loss: {sum(losses)/len(losses)}")
    return model_sample.cpu()


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


def setup_mnist():
    dist = get_approx_mnist_distribution()
    net = ConvEnergyModel((1, 8, 8), 3, 25)
    return dist, net


def main():
    dist, net = setup_mnist()
    samples = dist.rvs(100)
    print(samples.shape)
    dataset = data.TensorDataset(torch.tensor(samples, dtype=torch.float))
    energy = lambda x: net(torch.tensor(x, dtype=torch.float)).detach().numpy()
    train_more = "y"
    model_sample = None
    while train_more == "y":
        model_sample = train(net, dataset, num_steps=1, model_sample=model_sample)
        dist.visualize(plt.gcf(), model_sample, energy)
        plt.show()
        train_more = input("Train more?")


if __name__ == '__main__':
    main()
