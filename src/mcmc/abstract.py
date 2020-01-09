from typing import TYPE_CHECKING

import torch
from toolz import curry

if TYPE_CHECKING:
    from src.model import BaseEnergyModel


class MCSampler:
    def __call__(
        self, net: "BaseEnergyModel", x: torch.Tensor, beta=None
    ) -> torch.Tensor:
        """Perform a single MC step."""
        raise NotImplementedError

    @curry
    def log_metrics(self, logger, global_step: int, **kwargs):
        """Log any metrics to the tb_logger"""
        pass
