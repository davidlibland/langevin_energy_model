from typing import TYPE_CHECKING
from typing import Callable

import torch
from toolz import curry

import energy_model.utils.math
from energy_model.mcmc.abstract import MCSampler

if TYPE_CHECKING:
    from energy_model.model import BaseEnergyModel

UPPER_ACCEPTANCE_BOUND = 0.7
LOWER_ACCEPTANCE_BOUND = 0.4
LR_ADJUSTMENT = 1.0001


class MALASampler(MCSampler):
    def __init__(self, lr, beta=None, logger: Callable = None):
        self.lr = lr
        self.acceptance_ratio = 0.5
        self.beta = 1 if beta is None else beta
        self.logger = logger if logger is not None else lambda *args, **kwargs: None

    def __call__(
        self, net: "BaseEnergyModel", x: torch.Tensor, beta=None
    ) -> torch.Tensor:
        """Perform a single metropolis adjusted langevin MC update."""
        if beta is None:
            beta = self.beta

        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.data.zero_()
        y = net(x, beta=beta) / (beta + 1)
        x.retain_grad()
        y.sum().backward()
        grad_x = x.grad

        avg_energy_grad = float(energy_model.utils.math.avg_norm(grad_x))
        self.logger(avg_energy_grad=avg_energy_grad)

        # Hack to keep gradients in control:
        lr = self.lr / max(1, float(grad_x.abs().max()))

        noise_scale = torch.sqrt(torch.as_tensor(lr * 2) / (beta + 1))
        x_det = (x - lr * grad_x).detach()
        noise_f = noise_scale * torch.randn_like(x)
        x_ = x_det + noise_f

        self.logger(energy_grad_to_noise=avg_energy_grad * lr / float(noise_scale))

        log_q_x_x = -(noise_f ** 2).sum(dim=net.feature_dims) / (4 * lr)

        x_.requires_grad_(True)
        if x_.grad is not None:
            x_.grad.data.zero_()
        y_ = net(x_, beta=beta) / (beta + 1)
        y_.sum().backward()
        grad_x_ = x_.grad

        eps = ((x - x_ + lr * grad_x_) ** 2).sum(dim=net.feature_dims)
        log_q_xx_ = -eps / (4 * lr)

        log_alpha = y - y_ + log_q_xx_ - log_q_x_x
        alpha = torch.exp(torch.clamp_max(log_alpha, 0))
        mask = torch.rand(x.shape[0], device=alpha.device) < alpha
        # adjust the learning rate based on the acceptance ratio:
        acceptance_ratio = torch.mean(mask.float()).float()
        if acceptance_ratio.float() < LOWER_ACCEPTANCE_BOUND:
            self.lr /= LR_ADJUSTMENT
        elif acceptance_ratio.float() > UPPER_ACCEPTANCE_BOUND:
            self.lr *= LR_ADJUSTMENT
        self.acceptance_ratio = float(
            0.1 * acceptance_ratio + 0.9 * self.acceptance_ratio
        )
        self.logger(
            mala_lr=float(self.lr), mala_acceptance_ratio=float(acceptance_ratio)
        )
        while len(mask.shape) < len(x.shape):
            # Add extra feature-dims
            mask.unsqueeze_(dim=-1)
        result = torch.where(mask, x_, x).detach()
        avg_distance = energy_model.utils.math.avg_norm(result - x)
        self.logger(avg_sample_distance=float(avg_distance))
        return result

    @staticmethod
    def log_q(net, lr, x_, x, beta):
        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.data.zero_()
        y = net(x, beta=beta).sum()
        y.backward()
        grad_x = x.grad

        eps = ((x_ - x + lr * grad_x) ** 2).sum(dim=net.feature_dims)
        return -eps / (4 * lr)

    @curry
    def log_metrics(self, logger, global_step: int, **kwargs):
        """Log any metrics to the tb_logger"""
        logger(mala_lr=self.lr)
        logger(mala_acceptance_ratio=self.acceptance_ratio)

    def state_dict(self) -> dict:
        """Returns a dictionary of the complete state of the sampler"""
        return {
            "lr": self.lr,
            "beta": self.beta,
            "acceptance_ratio": self.acceptance_ratio,
        }

    def load_state_dict(self, state: dict):
        """Sets the state based on the dict supplied."""
        self.lr = state["lr"]
        self.beta = state["beta"]
        self.acceptance_ratio = state["acceptance_ratio"]
