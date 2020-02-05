from typing import TYPE_CHECKING

import torch

from src.mcmc.abstract import MCSampler

if TYPE_CHECKING:
    from src.model import BaseEnergyModel


class LangevinSampler(MCSampler):
    """
    This sampler implements Langevin Dynamics
    (https://en.wikipedia.org/wiki/Langevin_dynamics)
    """

    def __init__(self, lr):
        self.lr = lr

    def __call__(
        self, net: "BaseEnergyModel", x: torch.Tensor, beta=None
    ) -> torch.Tensor:
        """Perform a single langevin MC update."""
        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.data.zero_()
        y = net(x, beta=beta).sum()
        x.retain_grad()
        y.backward()
        grad_x = x.grad

        # Hack to keep gradients in control:
        lr = self.lr / max(1, float(grad_x.abs().max()))

        noise_scale = torch.sqrt(torch.as_tensor(lr * 2))
        result = x - lr * grad_x + noise_scale * torch.randn_like(x)
        return result.detach()
