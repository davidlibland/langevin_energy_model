from functools import lru_cache

import numpy as np
from sklearn.datasets import fetch_openml

from distributions.utils import train_gmm_pca_model, plot_image_samples, \
    get_samples
from .core import Distribution

@lru_cache()
def get_approx_mnist_distribution(n_pca_comp=10, n_mixtures=5,
                                  covariance_type="spherical") -> Distribution:
    """
    Returns an GMM approximation to MNIST.

    Args:
        n_pca_comp: The number of dimensions to reduce the image size to.
        n_mixtures: The number of mixtures to use for each digit.

    Returns:
        Distribution: An approximate model of mnist.
    """
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X/X.max()
    y = np.array([int(v) for v in y])
    distributions = []

    for i in range(10):
        print(f"training model on digit {i}")
        dist = train_gmm_pca_model(X[y==i, :],
                                       n_mixtures=n_mixtures,
                                       n_pca_comp=n_pca_comp,
                                   covariance_type=covariance_type)
        distributions.append(dist)
    mnist_dist = Distribution.mixture(distributions)
    mnist_dist.visualize = plot_image_samples([28, 28])
    mnist_dist.rvs = get_samples(X)
    return mnist_dist
