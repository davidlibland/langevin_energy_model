from collections import defaultdict
from dataclasses import dataclass
from typing import Callable
from typing import Tuple

import torch

import numpy as np

import utils.beta_schedules
from mcmc.mala import MALASampler
from model import BaseEnergyModel
from training_loop import CheckpointCallback


@dataclass
class AISState:
    log_w0: torch.Tensor
    log_p0: torch.Tensor
    current_samples: torch.Tensor


class AISLoss(CheckpointCallback):
    def __init__(self, logger: Callable, beta_schedule=None,
                 num_chains: int=20, num_mc_steps=1,
                 log_z_update_interval=5, device=None,
                 mc_dynamics=None):
        if beta_schedule is None:
            beta_schedule = utils.beta_schedules.build_schedule(
                ("arith", .01, 200),
                ("geom", 1., 1000)
            )
        self.beta_schedule = beta_schedule
        self.num_chains = num_chains
        self.logger = logger
        self.log_z_update_interval = log_z_update_interval
        self.num_mc_steps = num_mc_steps
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._sample_state = None
        if mc_dynamics is None:
            mc_dynamics = MALASampler(lr=0.1)
        self.mc_dynamics = mc_dynamics

    def update_log_z(self, net: BaseEnergyModel):
        """Update the estimate of log_Z"""
        net.eval()
        current_samples = net.sample_from_prior(self.num_chains, device=self.device)
        log_w = net(current_samples, beta=0).detach()
        for beta in self.beta_schedule[1:-1]:
            log_w -= net(current_samples, beta=beta).detach()
            current_samples = net.sample_fantasy(current_samples,
                                                 num_mc_steps=self.num_mc_steps,
                                                 beta=beta,
                                                 mc_dynamics=self.mc_dynamics
                                                 ).detach()
            log_w += net(current_samples, beta=beta).detach()
        log_w -= net(current_samples, beta=self.beta_schedule[-1]).detach()

        # Store the weights:
        self._sample_state = AISState(
            log_w0=log_w.detach(),
            log_p0=-net(current_samples).detach(),
            current_samples=current_samples.detach(),
        )

    def __call__(self, net: BaseEnergyModel, data_sample, global_step, **kwargs):
        if self._sample_state is None or global_step % self.log_z_update_interval == 0:
            self.update_log_z(net)

        # compute log_z
        net.eval()
        # get the correction to the log_weights:
        log_p1 = -net(self._sample_state.current_samples)
        log_w = log_p1 - self._sample_state.log_p0 + self._sample_state.log_w0
        num_chains = log_w.shape[0]
        log_z = torch.logsumexp(log_w, dim=0) - np.log(num_chains)

        loss = float(net(data_sample).mean().cpu()+log_z)

        # log the loss:

        # get the diagnostics
        log_w_var, effective_sample_size = self.get_diagnostic_stats(log_w)
        self.logger(loss_ais=loss,
                    ais_log_w_var=log_w_var,
                    ais_effective_sample_size=effective_sample_size
                    )

    @staticmethod
    def get_diagnostic_stats(log_w) -> Tuple[float, float]:
        """
        Returns diagnostic stats for a set of log weights.

        Args:
            log_w: The weights to diagnose

        Returns:
            Tuple: log variance of the weights, the effective sample size
        """
        num_chains = log_w.shape[0]
        log_w_var = torch.var(log_w).cpu()
        exp_std = torch.exp(log_w_var)
        if exp_std == 0:
            effective_sample_size = float("inf")
        elif not torch.isfinite(exp_std):
            effective_sample_size = 0
        else:
            effective_sample_size = float(num_chains / exp_std.cpu())
        return float(log_w_var), effective_sample_size
