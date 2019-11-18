from collections import defaultdict
from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from mcmc.mala import MALASampler
from model import BaseEnergyModel
from training_loop import CheckpointCallback


class AISLoss(CheckpointCallback):
    def __init__(self, tb_writer: SummaryWriter, beta_schedule=None,
                 num_chains: int=20, num_mc_steps=1,
                 log_z_update_interval=5, device=None,
                 mc_dynamics=None):
        if beta_schedule is None:
            beta_schedule = self.build_schedule(
                ("arith", .01, 60),
                ("geom", 1., 300)
            )
        self.beta_schedule = beta_schedule
        self.num_chains = num_chains
        self.tb_writer = tb_writer
        self.log_z_update_interval = log_z_update_interval
        self.num_mc_steps = num_mc_steps
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.log_z = None
        if mc_dynamics is None:
            mc_dynamics = MALASampler(lr=0.1)
        self.mc_dynamics = mc_dynamics

    def update_log_z(self, net: BaseEnergyModel):
        """Update the estimate of log_Z"""
        net.eval()
        current_samples = net.sample_from_prior(self.num_chains, device=self.device)
        log_w = net(current_samples, beta=0)
        for beta in self.beta_schedule[1:-1]:
            log_w -= net(current_samples, beta=beta)
            current_samples = net.sample_fantasy(current_samples,
                                                 num_mc_steps=self.num_mc_steps,
                                                 beta=beta,
                                                 mc_dynamics=self.mc_dynamics
                                                 )
            log_w += net(current_samples, beta=beta)
        log_w -= net(current_samples, beta=self.beta_schedule[-1])

        self.log_z = torch.logsumexp(log_w, dim=0) - np.log(self.num_chains)

        # Compute diagnostic stats:
        self.log_w = log_w
        self.log_w_var = torch.var(self.log_w).cpu()
        exp_std = torch.exp(self.log_w_var)
        if exp_std == 0:
            self.effective_sample_size = float("inf")
        elif not torch.isfinite(exp_std):
            self.effective_sample_size = 0
        else:
            self.effective_sample_size = float(self.num_chains / exp_std.cpu())

    def __call__(self, net: BaseEnergyModel, data_sample, global_step, **kwargs):
        if global_step % self.log_z_update_interval == 0:
            self.update_log_z(net)
            self.tb_writer.add_scalar("ais/log_w_var", scalar_value=self.log_w_var.cpu(), global_step=global_step)
            self.tb_writer.add_scalar("ais/effective_sample_size", scalar_value=self.effective_sample_size, global_step=global_step)
            self.tb_writer.add_histogram("ais/log_w", values=self.log_w.cpu(), global_step=global_step)
        if self.log_z is None:
            return
        net.eval()
        loss = float(net(data_sample).mean().cpu()+self.log_z)
        self.tb_writer.add_scalar("loss/ais", scalar_value=loss, global_step=global_step)


    @staticmethod
    def build_schedule(*instructions: Tuple[str, float, int]) -> np.ndarray:
        """
        Parses instructions of the form:
            ("geom", 0.1, 5), ("arith", 0.5, 10), ..., ("geom", 1, 6)
        To produce a geometric sequence of length 5 (from 0 upto .1) followed by
        an arithmetic sequence of length 10 (upto 0.5), ... followed by
        a geometric sequence of length 6 (upto 1)

        Note: Radford Neal suggests using a geometric schedule (but sometimes
        flattens it near zero to an arithmetic schedule). AIS is most
        computationally efficient when the schedule length is equal to the
        variance of the log-weights (which is one of the scalar metrics output
        by this Metric).
        (cf. https://www.cs.toronto.edu/~radford/ftp/ais-rev.pdf)

        Note: The first instruction must be arithmetic (to hit zero), while the
        last instruction must end at 1.

        Args:
            instructions: The instructions used to build the schedule.

        Returns:
            np.ndarray (the schedule).
        """
        sequence = []
        start = 0.
        stop = 0
        for seq_type, stop, num in instructions:
            assert 0 < stop <= 1, f"Invalid input: {seq_type, stop, num} " \
                                  f"stops must lie in the half open interval."
            if seq_type.lower()[0] == "a":
                # Arithmetic case
                seq = np.linspace(start=start, stop=stop, num=num)
            elif seq_type.lower()[0] == "g":
                # Geometric case
                if start == 0:
                    raise ValueError("Instructions must start with an "
                                     "arithmetic sequence")
                seq = np.exp(np.linspace(start=np.log(start),
                                         stop=np.log(stop), num=num))
            else:
                raise ValueError(f"Unrecognized sequence type: {seq_type}")
            sequence.append(seq)
            start = stop
        assert stop == 1, "The instructions must end at 1."
        schedule = np.concatenate(sequence).flatten()
        schedule[-1] = 1  # Ensure it ends at 1 (despite numerical errors).
        return schedule

