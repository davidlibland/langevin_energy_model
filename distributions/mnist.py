from functools import lru_cache
from math import sqrt

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml


from .core import Distribution, Normal, ApplyTransform

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
    mnist_dist.visualize = plot_model_samples
    return mnist_dist


def train_gmm_pca_model(X, n_pca_comp=10, n_mixtures=5,
                        covariance_type="spherical") -> Distribution:
    """
    Returns a GMM approximating the leading components of X.

    Args:
        X (tensor ~ (n_samples, n_features): The data to model.
        n_pca_comp: The number of dimensions to reduce the image size to.
        n_mixtures: The number of mixtures to use for each digit.

    Returns:
        Distribution modeling the leading PCA components of X.
    """
    dec = PCA(n_components=n_pca_comp, whiten=True)
    clf = GaussianMixture(n_components=n_mixtures, covariance_type=covariance_type)
    dec.fit(X)
    X_ = dec.transform(X)
    clf.fit(X_)

    def trans(arr: np.ndarray) -> np.ndarray:
        tr_arr = dec.inverse_transform(arr)
        return tr_arr

    def inv_trans(arr: np.ndarray) -> np.ndarray:
        tr_arr = dec.transform(arr)
        return tr_arr

    dist = Normal.from_sklearn_gmm(clf)
    trans_dist = ApplyTransform(dist=dist,
                                trans=trans,
                                inv_trans=inv_trans)
    return trans_dist


def plot_model_samples(fig: plt.Figure, X: np.ndarray, binarize=False):
    """Plots model samples on the given figure."""
    fig.clear()
    if binarize:
        X = X > .5
    n = int(sqrt(X.shape[0]))

    ax = fig.subplots(ncols=n, nrows=n)
    for i in range(n):
        for j in range(n):
            ax[i, j].imshow(X[i+n*j, :].reshape([28, 28]),
                           cmap='gray', vmin=0, vmax=1)
            ax[i, j].axis("off")