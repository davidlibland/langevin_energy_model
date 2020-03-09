"""
Mathematical utility functions
"""
import torch


def avg_norm(tensor: torch.Tensor):
    """Computes the average tensor norm."""
    square = (tensor**2).sum(dim=tuple(range(1, tensor.ndim)))
    return (square**(0.5)).mean()


def identity(x):
    """The identity function."""
    return x


def swish(x):
    """The swish activation function."""
    return x*torch.sigmoid(x)