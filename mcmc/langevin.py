import torch

from mcmc.abstract import MCSampler


class LangevinSampler(MCSampler):
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, net: "BaseEnergyModel", x: torch.Tensor, beta=None) -> torch.Tensor:
        """Perform a single langevin MC update."""
        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.data.zero_()
        y = net(x, beta=beta).sum()
        y.backward()
        grad_x = x.grad

        # Hack to keep gradients in control:
        lr = self.lr/max(1, float(grad_x.abs().max()))

        noise_scale = torch.sqrt(torch.as_tensor(lr*2))
        result = x - lr*grad_x+noise_scale*torch.randn_like(x)
        return result.detach()
