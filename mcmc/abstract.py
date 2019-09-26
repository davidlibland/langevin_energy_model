import torch


class MCSampler:
    def __call__(self, net: "BaseEnergyModel", x: torch.Tensor, beta=None) -> torch.Tensor:
        """Perform a single MC step."""
        raise NotImplementedError

    def log_metrics(self, tb_writer, global_step: int, **kwargs):
        """Log any metrics to the tb_logger"""
        pass