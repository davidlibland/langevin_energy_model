from typing import TYPE_CHECKING

import torch
from toolz import curry

from src.mcmc.abstract import MCSampler

if TYPE_CHECKING:
    from src.model import BaseEnergyModel

UPPER_ACCEPTANCE_BOUND = 0.7
LOWER_ACCEPTANCE_BOUND = 0.4
LR_ADJUSTMENT = 1.0001


class MALASampler(MCSampler):
    def __init__(self, lr):
        self.lr = lr
        self.acceptance_ratio = 0.5

    def __call__(
        self, net: "BaseEnergyModel", x: torch.Tensor, beta=None
    ) -> torch.Tensor:
        """Perform a single metropolis adjusted langevin MC update."""
        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.data.zero_()
        y = net(x, beta=beta)
        x.retain_grad()
        y.sum().backward()
        grad_x = x.grad

        # Hack to keep gradients in control:
        lr = self.lr / max(1, float(grad_x.abs().max()))

        noise_scale = torch.sqrt(torch.as_tensor(lr * 2))
        x_det = (x - lr * grad_x).detach()
        noise_f = noise_scale * torch.randn_like(x)
        x_ = x_det + noise_f

        log_q_x_x = -(noise_f ** 2).sum(dim=1, keepdim=True) / (4 * lr)

        x_.requires_grad_(True)
        if x_.grad is not None:
            x_.grad.data.zero_()
        y_ = net(x_, beta=beta)
        y_.sum().backward()
        grad_x_ = x_.grad

        eps = ((x - x_ + lr * grad_x_) ** 2).sum(dim=1, keepdim=True)
        log_q_xx_ = -eps / (4 * lr)

        log_alpha = y - y_ + log_q_xx_ - log_q_x_x
        alpha = torch.exp(torch.clamp_max(log_alpha, 0))
        mask = torch.rand(x.shape[0], 1, device=alpha.device) < alpha
        # adjust the learning rate based on the acceptance ratio:
        acceptance_ratio = torch.mean(mask.float()).float()
        if acceptance_ratio.float() < LOWER_ACCEPTANCE_BOUND:
            self.lr /= LR_ADJUSTMENT
        elif acceptance_ratio.float() > UPPER_ACCEPTANCE_BOUND:
            self.lr *= LR_ADJUSTMENT
        self.acceptance_ratio = float(
            0.1 * acceptance_ratio + 0.9 * self.acceptance_ratio
        )
        return torch.where(mask, x_, x).detach()

    @staticmethod
    def log_q(net, lr, x_, x, beta):
        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.data.zero_()
        y = net(x, beta=beta).sum()
        y.backward()
        grad_x = x.grad

        eps = ((x_ - x + lr * grad_x) ** 2).sum(dim=1, keepdim=True)
        return -eps / (4 * lr)

    @curry
    def log_metrics(self, logger, global_step: int, **kwargs):
        """Log any metrics to the tb_logger"""
        logger(mala_lr=self.lr)
        logger(mala_acceptance_ratio=self.acceptance_ratio)