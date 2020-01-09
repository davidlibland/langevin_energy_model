from functools import lru_cache
from typing import Tuple

import numpy as np
from torch.utils.data import Sampler

from src.distributions.utils import plot_image_samples
from .core import Sampler

PATTERNS = {
    "checkerboard_2x2": np.array([[0, 1], [1, 0]], dtype=np.float32),
    "diagonal_gradient_2x2": np.array([[0, 0.5], [0.5, 1]], dtype=np.float32),
}


@lru_cache()
def get_pattern_distribution(
    patterns: Tuple[str, ...] = ("checkerboard",)
) -> "Sampler":
    """
    Returns a pattern sampler.

    Args:
        patterns: A tuple of patterns to use.

    Returns:
        Sampler: Sampling from the patterns
    """
    pattern_arrays = [PATTERNS[pattern].flatten() for pattern in patterns]
    X = np.stack(pattern_arrays, axis=0)
    pattern_dist = Sampler.from_samples(X)
    pattern_dist.visualize = plot_image_samples([2, 2], False)
    return pattern_dist
