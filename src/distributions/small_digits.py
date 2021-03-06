from functools import lru_cache

import numpy as np
import torch
from sklearn.datasets import load_digits

from src.distributions.core import Sampler, Distribution
from src.distributions.utils import train_gmm_pca_model, plot_image_samples, get_samples


@lru_cache()
def get_sm_digit_distribution() -> Sampler:
    """
    Returns a Digits Sampler.

    Returns:
        Sampler: Sampling digits
    """
    X, y = load_digits(return_X_y=True)
    X = 2 * X.astype(np.float) / X.max() - 1
    n = X.shape[0]
    X_torch = torch.tensor(X.reshape([n, 1, 8, 8]))
    X = torch.nn.AvgPool2d(kernel_size=(2, 2))(X_torch).numpy()
    digit_dist = Sampler.from_samples(
        X, noise=lambda shape: 2 * np.random.rand(*shape) / 255
    )
    digit_dist.visualize = plot_image_samples([4, 4], False)
    return digit_dist


@lru_cache()
def get_approx_sm_digit_distribution(
    n_pca_comp=10, n_mixtures=5, covariance_type="spherical"
) -> Distribution:
    """
    Returns an GMM approximation to MNIST.

    Args:
        n_pca_comp: The number of dimensions to reduce the image size to.
        n_mixtures: The number of mixtures to use for each digit.

    Returns:
        Distribution: An approximate model of mnist.
    """
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float) / X.max()
    print(X.shape)
    n = X.shape[0]
    X_torch = torch.tensor(X.reshape([n, 8, 8]))
    X = torch.nn.AvgPool2d(kernel_size=(2, 2))(X_torch).numpy().reshape([n, -1])
    y = np.array([int(v) for v in y])
    distributions = []

    for i in range(10):
        print(f"training model on digit {i}")
        dist = train_gmm_pca_model(
            X[y == i, :],
            n_mixtures=n_mixtures,
            n_pca_comp=n_pca_comp,
            covariance_type=covariance_type,
        )
        distributions.append(dist)
    mnist_dist = Distribution.mixture(distributions)
    mnist_dist.visualize = plot_image_samples([4, 4], False)
    mnist_dist.rvs = get_samples(X)
    return mnist_dist
