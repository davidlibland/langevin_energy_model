"""
Standard distributions to model.
"""
from abc import ABC, abstractmethod
from math import pi, sqrt
from typing import Optional, List, Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

plt.switch_backend("agg")


from src.distributions.utils import get_samples


class Sampler(ABC):
    @staticmethod
    def from_samples(X: np.ndarray, noise=None) -> "Sampler":
        sampler = Sampler()
        sampler.rvs = get_samples(X, noise=noise)
        return sampler

    def rvs(self, size: int) -> np.ndarray:
        """
        Sample from the distribution.

        Args:
            size (int): The number of samples to draw.

        Returns:
            tensor ~ (size, n_features)
        """
        raise NotImplementedError

    def visualize(
        self,
        fig: plt.Figure,
        X: np.ndarray,
        energy: Callable[[np.ndarray], np.ndarray] = None,
    ):
        """
        Visualize a set of samples, returning a figure.

        Args:
            X (tensor ~ (n_samples, n_features):

        Returns:
            Figure
        """
        fig.clear()
        ax = fig.subplots(1, 1)
        if X.shape[1] == 1:
            ax.hist(X.reshape([-1]), density=True, label="Data")
            x_min = X.min()
            x_max = X.max()
            xs = np.linspace(x_min, x_max, 100)
            if hasattr(self, "logpdf"):
                ys_ = np.exp(self.logpdf(xs.reshape([-1, 1])))
                ys = ys_.reshape(-1)
                ax.plot(xs, ys, label="Actual")
            if energy is not None:
                ys_ = np.exp(-energy(xs.reshape([-1, 1])))
                ys = ys_.reshape(-1)
                Z = ys_.mean() * (x_max - x_min)
                ax.plot(xs, ys / Z, label="Energy", color="red")
            ax.legend()
        elif X.shape[1] == 2:
            ax.scatter(X[:, 0], X[:, 1], label="Data")
            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            y_min, y_max = X[:, 1].min(), X[:, 1].max()
            x_support = np.linspace(x_min, x_max, 100)
            y_support = np.linspace(y_min, y_max, 100)
            xx, yy = np.meshgrid(x_support, y_support)
            XY = np.hstack([xx.reshape([-1, 1]), yy.reshape([-1, 1])])
            if hasattr(self, "logpdf"):
                z_ = np.exp(self.logpdf(XY))
                z = z_.reshape(xx.shape)
                ax.contour(xx, yy, z, 10)
            if energy is not None:
                z_ = np.exp(-energy(XY))
                z = z_.reshape(xx.shape)
                ax.contour(xx, yy, z, 10, cmap="Reds")
            ax.legend()
        else:
            from sklearn.manifold import TSNE

            tsne = TSNE(n_components=2)
            emb = tsne.fit_transform(X)
            ax.scatter(emb[:, 0], emb[:, 1])


class Distribution(Sampler):
    @abstractmethod
    def logpdf(self, X) -> np.ndarray:
        """
        The log pdf of values X.

        Args:
            X (tensor ~ (n_samples, n_features)): The values at which to compute
                the pdf

        Returns:
            tensor ~ (n_samples,)
        """
        raise NotImplementedError

    @staticmethod
    def mixture(
        components: List["Distribution"], weights: Optional[List[float]] = None
    ):
        """
        Forms a mixture distribution from the components provided.

        Args:
            components (List[Distribution]): The components of the mixture.
            weights (List[float], optional): The weights of the components.
        """
        return MixtureDistribution(components=components, weights=weights)


class MixtureDistribution(Distribution):
    def __init__(
        self, components: List[Distribution], weights: Optional[List[float]] = None
    ):
        """
        Forms a mixture distribution from the components provided.

        Args:
            components (List[Distribution]): The components of the mixture.
            weights (List[float], optional): The weights of the components.
        """
        self.n_components = len(components)
        if weights is None:
            weights = [1 / self.n_components for _ in range(self.n_components)]
        self.weights = weights
        self.components = components

    @property
    def coefficients(self):
        return np.array(self.weights)

    def logpdf(self, X) -> np.ndarray:
        """
        The log pdf of values X.

        Args:
            X (tensor ~ (n_samples, n_features)): The values at which to compute
                the pdf

        Returns:
            tensor ~ (n_samples,)
        """
        comp_logpdfs = [np.reshape(comp.logpdf(X), [-1, 1]) for comp in self.components]
        logpdfs = np.hstack(comp_logpdfs)
        result = logsumexp(logpdfs, axis=1, b=self.coefficients)
        return np.array(result)

    def rvs(self, size: int) -> np.ndarray:
        """
        Sample from the distribution.

        Args:
            size (int): The number of samples to draw.

        Returns:
            tensor ~ (size, n_features)
        """
        comps = np.random.choice(a=self.components, p=self.coefficients, size=size)
        samples = [comp.rvs(size=1) for comp in comps]
        return np.vstack(samples)


class Normal(Distribution):
    def __init__(self, means: np.ndarray = None, scales: np.ndarray = None):
        if means is None:
            means = np.zeros((1,))
        if scales is None:
            scales = np.ones_like(means)
        assert means.shape == scales.shape
        assert len(means.shape) == 1
        self.ndim = means.shape[0]
        self.means = np.reshape(means, [1, -1])
        self.scales = np.reshape(scales, [1, -1])

    def logpdf(self, X) -> np.ndarray:
        """
        The log pdf of values X.

        Args:
            X (tensor ~ (n_samples, n_features)): The values at which to compute
                the pdf

        Returns:
            tensor ~ (n_samples,)
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        H = 0.5 * ((X - self.means) / self.scales) ** 2
        Z = sqrt(2 * pi) * self.scales
        return -np.sum(H + np.log(Z), axis=1)

    def rvs(self, size: int) -> np.ndarray:
        """
        Sample from the distribution.

        Args:
            size (int): The number of samples to draw.

        Returns:
            tensor ~ (size, n_features)
        """
        return np.random.randn(size, self.ndim) * self.scales + self.means

    @staticmethod
    def from_sklearn_gmm(gmm: GaussianMixture):
        """Returns a mixture of normals for diagonal mixtures"""
        components = []
        assert gmm.covariance_type in {"diag", "spherical"}
        if gmm.covariance_type == "diag":
            scale_getter = lambda std_ar: std_ar
        elif gmm.covariance_type == "spherical":
            scale_getter = lambda v: np.array([v for _ in range(gmm.means_.shape[1])])
        for mean_ar, std_ar in zip(gmm.means_, gmm.covariances_):
            scales = scale_getter(std_ar)
            mean = np.array(mean_ar)
            components.append(Normal(means=mean, scales=scales))
        return Distribution.mixture(components=components, weights=gmm.weights_)


class ApplyTransform(Distribution):
    def __init__(
        self,
        dist: Distribution,
        trans: Callable[[np.ndarray], np.ndarray],
        inv_trans: Callable[[np.ndarray], np.ndarray],
    ):
        """
        Transforms a distribution by provided transformations. Technically,
        this will no longer be a distribution unless these transformations
        are isometries.

        Args:
            dist (Distribution): The distribution to transform
            trans: The transformation to apply
            inv_trans: The inverse of trans.
        """
        self.dist = dist
        self.trans = trans
        self.inv_trans = inv_trans

    def logpdf(self, X) -> np.ndarray:
        """
        The log pdf of values X.

        Args:
            X (tensor ~ (n_samples, n_features)): The values at which to compute
                the pdf

        Returns:
            tensor ~ (n_samples,)
        """
        return self.dist.logpdf(self.inv_trans(X))

    def rvs(self, size: int) -> np.ndarray:
        """
        Sample from the distribution.

        Args:
            size (int): The number of samples to draw.

        Returns:
            tensor ~ (size, n_features)
        """
        return self.trans(self.dist.rvs(size))
