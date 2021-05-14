from typing import TYPE_CHECKING
from typing import Callable

import torch

import energy_model.utils.math
from energy_model.mcmc.abstract import MCSampler

if TYPE_CHECKING:
    from energy_model.model import BaseEnergyModel


class LangevinSampler(MCSampler):
    """
    This sampler implements Langevin Dynamics
    (https://en.wikipedia.org/wiki/Langevin_dynamics)
    """

    def __init__(self, lr, beta=None, logger: Callable = None):
        self.lr = lr
        self.logger = logger if logger is not None else lambda *args, **kwargs: None
        self.beta = 1 if beta is None else beta

    def __call__(
        self, net: "BaseEnergyModel", x: torch.Tensor, beta=None
    ) -> torch.Tensor:
        """Perform a single langevin MC update."""
        if beta is None:
            beta = self.beta
        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.data.zero_()
        y = net(x, beta=beta).sum() / (beta + 1)
        x.retain_grad()
        y.backward()
        grad_x = x.grad
        avg_energy_grad = float(grad_x.norm() / grad_x.shape[0])
        self.logger(avg_energy_grad=avg_energy_grad)

        # Hack to keep gradients in control:
        lr = self.lr / max(1, float(grad_x.abs().max()))

        noise_scale = torch.sqrt(torch.as_tensor(lr * 2) / (beta + 1))
        result = x - lr * grad_x + noise_scale * torch.randn_like(x)

        self.logger(energy_grad_to_noise=avg_energy_grad * lr / float(noise_scale))
        avg_distance = energy_model.utils.math.avg_norm(result - x)
        self.logger(avg_sample_distance=float(avg_distance))
        return result.detach()

    def state_dict(self) -> dict:
        """Returns a dictionary of the complete state of the sampler"""
        return {"lr": self.lr, "beta": self.beta}

    def load_state_dict(self, state: dict):
        """Sets the state based on the dict supplied."""
        self.lr = state["lr"]
        self.beta = state["beta"]
