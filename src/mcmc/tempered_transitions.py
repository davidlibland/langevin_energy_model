from typing import TYPE_CHECKING
from typing import Callable

import torch

import src.mcmc.abstract
import src.utils.beta_schedules

if TYPE_CHECKING:
    from src.model import BaseEnergyModel


class TemperedTransitions(src.mcmc.abstract.MCSampler):
    def __init__(
        self,
        mc_dynamics: src.mcmc.abstract.MCSampler,
        beta_schedule=None,
        logger: Callable = None,
    ):
        if beta_schedule is None:
            beta_schedule = src.utils.beta_schedules.build_schedule(
                ("geom", 1.0, 30), start=0.1,
            )
        self.beta_schedule = beta_schedule
        self.mc_dynamics = mc_dynamics
        self.num_mc_steps = 1
        self.logger = logger if logger is not None else lambda *args, **kwargs: None

    def __call__(
        self, net: "BaseEnergyModel", x: torch.Tensor, beta=None
    ) -> torch.Tensor:
        """Perform a single MC step."""
        del beta
        net.eval()
        current_samples = x
        # at beta=1
        log_alpha = net(current_samples, beta=self.beta_schedule[-1]).detach()
        for beta in reversed(self.beta_schedule[1:-1]):
            log_alpha -= net(current_samples, beta=beta).detach()
            current_samples = net.sample_fantasy(
                current_samples,
                num_mc_steps=self.num_mc_steps,
                beta=beta,
                mc_dynamics=self.mc_dynamics,
            ).detach()
            log_alpha += net(current_samples, beta=beta).detach()
        # at beta=0
        log_alpha -= net(current_samples, beta=self.beta_schedule[0]).detach()
        current_samples = net.sample_fantasy(
            current_samples,
            num_mc_steps=2 * self.num_mc_steps,
            beta=self.beta_schedule[0],
            mc_dynamics=self.mc_dynamics,
        ).detach()
        log_alpha += net(current_samples, beta=self.beta_schedule[0]).detach()
        for beta in self.beta_schedule[1:-1]:
            log_alpha -= net(current_samples, beta=beta).detach()
            current_samples = net.sample_fantasy(
                current_samples,
                num_mc_steps=self.num_mc_steps,
                beta=beta,
                mc_dynamics=self.mc_dynamics,
            ).detach()
            log_alpha += net(current_samples, beta=beta).detach()
        log_alpha -= net(current_samples, beta=self.beta_schedule[-1]).detach()

        alpha = torch.exp(torch.clamp_max(log_alpha, 0))
        mask = torch.rand(x.shape[0], device=alpha.device) < alpha

        while len(mask.shape) < len(x.shape):
            # Add extra feature-dims
            mask.unsqueeze_(dim=-1)
        acceptance_ratio = torch.mean(mask.float()).float()
        self.logger(tempered_acceptance_ratio=float(acceptance_ratio))
        result = torch.where(mask, current_samples, x).detach()
        avg_distance = src.utils.math.avg_norm(result - x)
        self.logger(tempered_avg_sample_distance=float(avg_distance))
        return result

    def state_dict(self) -> dict:
        """Returns a dictionary of the complete state of the sampler"""
        return {
            "transition_state": self.mc_dynamics.state_dict(),
            "beta_schedule": self.beta_schedule,
        }

    def load_state_dict(self, state: dict):
        """Sets the state based on the dict supplied."""
        self.beta_schedule = state["beta_schedule"]
        self.mc_dynamics.load_state_dict(state["transition_state"])
