from math import sqrt
from typing import Tuple, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from toolz import curry

if TYPE_CHECKING:
    from energy_model.distributions.core import Distribution


def train_gmm_pca_model(
    X, n_pca_comp=10, n_mixtures=5, covariance_type="spherical"
) -> "Distribution":
    """
    Returns a GMM approximating the leading components of X.

    Args:
        X (tensor ~ (n_samples, n_features): The data to model.
        n_pca_comp: The number of dimensions to reduce the image size to.
        n_mixtures: The number of mixtures to use for each digit.

    Returns:
        Distribution modeling the leading PCA components of X.
    """
    from energy_model.distributions.core import Normal, ApplyTransform

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
    trans_dist = ApplyTransform(dist=dist, trans=trans, inv_trans=inv_trans)
    return trans_dist


@curry
def plot_image_samples(
    im_size: Tuple[int, ...], binarize, fig: plt.Figure, X: np.ndarray, energy=None
):
    """Plots model samples on the given figure."""
    fig.clear()
    if binarize:
        X = X > 0.5
    n = min(int(sqrt(X.shape[0])), 7)

    ax = fig.subplots(ncols=n, nrows=n)
    for i in range(n):
        for j in range(n):
            ax[i, j].imshow(
                X[i + n * j, :].reshape(im_size), cmap="gray", vmin=-1, vmax=1
            )
            ax[i, j].axis("off")


@curry
def get_samples(X: np.ndarray, size: int, noise=None) -> np.ndarray:
    n = X.shape[0]
    samples = np.random.choice(n, size=size, replace=True)
    if noise is None:
        return X[samples, ...]
    result = X[samples, ...]
    return result + noise(result.shape)
