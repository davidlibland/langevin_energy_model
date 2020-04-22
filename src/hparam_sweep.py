import os
import random
import time
from collections import deque
from typing import Any
from typing import Callable
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from ray.tune import Trainable
from torch import optim

import src.mcmc.langevin
import src.mcmc.mala
import src.mcmc.tempered_transitions
import src.utils.ais
import src.utils.beta_schedules
import src.utils.constraints
from src import model
from src.distributions import core
from src.utils.ais import AISLoss


plt.switch_backend("agg")


MAX_REPLAY = 0
REPLAY_PROB = 0.99


class Logger:
    """
    A simple container for logging a variety of metrics.
    """

    def __init__(self):
        self.logs = dict()

    def __call__(self, **kwargs: float):
        """Logs the metrics passed by keyword, these are stored until flush is called"""
        for k, v in kwargs.items():
            if k not in self.logs:
                self.logs[k] = []
            self.logs[k].append(v)

    def flush(self):
        """Clears the store of logs"""
        self.logs = dict()

    def means(self) -> dict:
        """Returns the means of the metrics"""
        return {k: 0 if not vs else sum(vs) / len(vs) for k, vs in self.logs.items()}


def get_energy_trainer(
    setup_dist: Callable[[Any], Tuple[core.Sampler, model.BaseEnergyModel]]
):
    """Returns a tune trainable for the distribution and model architecture"""

    class EnergyTrainer(Trainable):
        def _setup(self, config):
            self.logger_ = Logger()
            self.lr = config.get("lr", 1e-2)
            self.weight_decay = config.get("weight_decay", 1e-1)
            self.num_mc_steps = config.get("num_mc_steps", 1)

            # Reweight number of steps if tempering is used:
            num_tempered_transitions = config.get("num_tempered_transitions", 1)
            sampler_beta_schedule_num_steps = max(
                self.num_mc_steps // num_tempered_transitions, 1
            )
            if "tempered" in config.get("sampler", "mala"):
                self.num_mc_steps = num_tempered_transitions

            self.num_sample_mc_steps = config.get("num_sample_mc_steps", 1000)
            self.sample_beta = config.get("sample_beta", 1)
            self.batch_size = config.get("batch_size", 1024)
            self.sample_size = config.get("sample_size", 30000)
            sampler_lr = config.get("sampler_lr", 0.1)
            beta_target = config.get("sampler_beta_target", 1.0)
            sampler_beta_schedule = src.utils.beta_schedules.build_schedule(
                ("geom", beta_target, sampler_beta_schedule_num_steps,),
                start=config.get("sampler_beta_min", 0.1),
            )
            samplers = {
                "mala": src.mcmc.mala.MALASampler(
                    lr=sampler_lr, beta=beta_target, logger=self.logger_
                ),
                "langevin": src.mcmc.langevin.LangevinSampler(
                    lr=sampler_lr, beta=beta_target, logger=self.logger_,
                ),
                "tempered langevin": src.mcmc.tempered_transitions.TemperedTransitions(
                    mc_dynamics=src.mcmc.langevin.LangevinSampler(
                        lr=sampler_lr, logger=self.logger_
                    ),
                    beta_schedule=sampler_beta_schedule,
                    logger=self.logger_,
                ),
                "tempered mala": src.mcmc.tempered_transitions.TemperedTransitions(
                    mc_dynamics=src.mcmc.mala.MALASampler(
                        lr=sampler_lr, logger=self.logger_
                    ),
                    beta_schedule=sampler_beta_schedule,
                    logger=self.logger_,
                ),
            }
            self.sampler = samplers.get(config.get("sampler", "mala"))
            self.verbose = True

            self.dist, self.net_ = setup_dist(**config)
            self.samples = self.dist.rvs(self.sample_size)
            print(self.samples.shape)
            dataset = data.TensorDataset(torch.tensor(self.samples, dtype=torch.float))
            self.energy = (
                lambda x: self.net_(torch.tensor(x, dtype=torch.float).to(self.device))
                .detach()
                .cpu()
                .numpy()
            )
            self.optimizer_ = optim.Adam(
                self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

            self.dataloader = data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
            )
            # Determine the device type:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net_.to(self.device)
            print(f"training on {len(dataset)} samples using {self.device}.")
            self.global_step_ = 0
            self.epoch_ = 0
            self.model_samples_ = None

            self.ais_loss = AISLoss(
                logger=self.logger_,
                num_chains=config.get("ais_num_chains", 100),
                log_z_update_interval=config.get("ais_update_interval", 32),
                max_interpolants=config.get("ais_max_interpolants", 5000),
                num_interpolants=config.get("ais_num_interpolants", 500),
            )
            self.step_callbacks = [self.ais_loss]

            # Add replay buffer config:
            self.max_replay = config.get("max_replay", MAX_REPLAY)
            self.replay_prob = config.get("replay_prob", REPLAY_PROB)

        def get_data_and_model_samples(self):
            """Get a batch of data and model samples."""
            model_samples = (
                self.net_.sample_fantasy(
                    x=self.model_samples_[-1],
                    num_mc_steps=self.num_sample_mc_steps,
                    beta=self.sample_beta,
                    mc_dynamics=self.sampler,
                )
                .detach()
                .cpu()
                .numpy()
            )
            data_sample_ixs = torch.randint(
                0, self.samples.shape[0], size=(model_samples.shape[0],)
            )
            data_samples = self.samples[data_sample_ixs, ...]
            return data_samples, model_samples

        def save_images(self, samples, label=None, dir=""):
            """Saves sample images at the giving dir"""
            if label is None:
                label = self.global_step_
            fig = plt.figure()
            self.net_.eval()
            self.dist.visualize(fig, samples, self.energy)
            plot_fn = os.path.join(dir, f"samples_{label}.png")
            fig.savefig(plot_fn)
            plt.close(fig)

        def save_energy_plot(self, data_samples, model_samples, label=None, dir=""):
            """Saves sample images at the giving dir"""
            if label is None:
                label = self.global_step_
            fig = plt.figure()
            self.net_.eval()
            data_energies = self.energy(data_samples).flatten()
            model_energies = self.energy(model_samples).flatten()
            min_length = min(data_energies.size, model_energies.size)
            data_energies = data_energies[:min_length]
            model_energies = model_energies[:min_length]
            energies = np.stack([data_energies, model_energies], axis=1)
            energies -= energies.min()  # Normalize the energies.
            fig.clear()
            ax = fig.subplots(1, 1)
            ax.hist(energies, density=True, label=["data", "model"])
            ax.legend(prop={"size": 10})

            plot_fn = os.path.join(dir, f"energies_{label}.png")
            fig.savefig(plot_fn)
            plt.close(fig)

        def save_model(self, dir="", **kwargs):
            """Saves the model at the given dir"""
            ckpt_fn = os.path.join(dir, f"model.pkl")
            torch.save(
                {
                    "global_step": self.global_step_,
                    "epoch": self.epoch_,
                    "model": self.net_.state_dict(),
                    "optimizer": self.optimizer_.state_dict(),
                    "sampler_state": self.sampler.state_dict(),
                    "model_samples": list(self.model_samples_),
                    "ais_state": self.ais_loss.state_dict(),
                    "replay_prob": self.replay_prob,
                    "max_replay": self.max_replay,
                },
                ckpt_fn,
            )
            return ckpt_fn

        def load_model(self, ckpt_fn):
            """Loads the model from the specified dir."""
            checkpoint = torch.load(ckpt_fn)
            self.net_.load_state_dict(checkpoint["model"])
            self.optimizer_.load_state_dict(checkpoint["optimizer"])
            self.epoch_ = checkpoint["epoch"]
            self.global_step_ = checkpoint["global_step"]
            self.model_samples_ = deque(checkpoint["model_samples"])
            self.sampler.load_state_dict(checkpoint["sampler_state"])
            self.ais_loss.load_state_dict(checkpoint["ais_state"])
            self.replay_prob = checkpoint["replay_prob"]
            self.max_replay = checkpoint["max_replay"]

        def _train(self):
            """Train the model on one epoch"""
            epoch_training_time = 0
            epoch_metrics_time = 0
            self.epoch_ += 1
            for i_batch, sample_batched in enumerate(self.dataloader):
                self.global_step_ += 1
                batch_start_time = time.time()
                data_sample = sample_batched[0].to(self.device)

                # Get model samples, either from replay buffer or noise.
                if self.model_samples_ is None:
                    self.model_samples_ = deque(
                        [
                            self.net_.sample_from_prior(
                                data_sample.shape[0], device=self.device
                            ).detach()
                        ]
                    )
                elif len(self.model_samples_) > self.max_replay:
                    self.model_samples_.popleft()
                replay_sample = random.choices(
                    self.model_samples_,
                    # favor more recent samples:
                    weights=list(range(1, len(self.model_samples_) + 1)),
                )[0]
                noise_sample = self.net_.sample_from_prior(
                    replay_sample.shape[0], device=self.device
                )
                mask = torch.rand(replay_sample.shape[0]) < self.replay_prob
                while len(mask.shape) < len(replay_sample.shape):
                    # Add extra feature-dims
                    mask.unsqueeze_(dim=-1)

                model_sample = torch.where(
                    mask.to(self.device), replay_sample, noise_sample
                )

                self.net_.eval()
                # Run at least one iteration
                model_sample = self.net_.sample_fantasy(
                    model_sample,
                    num_mc_steps=self.num_mc_steps,
                    mc_dynamics=self.sampler,
                ).detach()

                self.model_samples_.append(model_sample)

                # Sanity checks:
                assert (
                    data_sample.shape[1:] == self.net_.input_shape
                ), "Data is incompatible with network."
                assert (
                    model_sample.shape[1:] == data_sample.shape[1:]
                ), "Model and data samples are incompatible."

                # Forward gradient:
                self.net_.train()
                self.net_.zero_grad()
                data_energy_mean = self.net_(data_sample).mean()
                model_energy = self.net_(model_sample)
                model_energy_mean = model_energy.mean()

                # Estimate the odds of the data's energy based on a normal fitted to
                # model samples:
                data_erf = torch.erf(
                    (data_energy_mean - model_energy_mean) / model_energy.std()
                )

                objective = data_energy_mean - model_energy_mean
                objective.backward()
                torch.nn.utils.clip_grad.clip_grad_value_(self.net_.parameters(), 1e2)
                self.optimizer_.step()

                batch_training_time = time.time() - batch_start_time
                epoch_training_time += batch_training_time
                self.logger_(energy_diff=float(objective))
                self.logger_(data_erf=float(data_erf))

                tr_metrics_start_time = time.time()
                for callback in self.step_callbacks:
                    callback(
                        net=self.net_,
                        data_sample=data_sample,
                        model_sample=model_sample,
                        epoch=self.epoch_,
                        global_step=self.global_step_,
                        validation=False,
                    )
                tr_metrics_time = time.time() - tr_metrics_start_time
                epoch_metrics_time += tr_metrics_time
                if self.verbose:
                    print(
                        f"on epoch {self.epoch_}, batch {i_batch}, data erf: {data_erf}, objective: {objective}"
                    )
                    print(f"model energy: {model_energy_mean} +- {model_energy.std()}")
                    print(f"data energy: {data_energy_mean}")
                    print(
                        f"training time: {batch_training_time:0.3f}s, metrics time: {tr_metrics_time:0.3f}s"
                    )
            means = self.logger_.means()
            if self.verbose:
                print(f"on epoch {self.epoch_}")
                for k, v in means.items():
                    print(f"{k}: {v}")
            self.logger_.flush()
            means["loss"] = src.utils.constraints.add_soft_constraint(
                means["loss_ais"], means["data_erf"], lower_bound=-1
            )
            return means

        def _save(self, tmp_checkpoint_dir):
            """Save the model"""
            neg_samples = self.model_samples_[-1].detach().cpu().numpy()
            data_samples, model_samples = self.get_data_and_model_samples()
            self.save_images(model_samples, dir=tmp_checkpoint_dir)
            self.save_images(
                neg_samples, dir=tmp_checkpoint_dir, label=f"_neg_{self.global_step_}"
            )
            self.save_images(
                data_samples, dir=tmp_checkpoint_dir, label=f"_data_{self.global_step_}"
            )
            self.save_energy_plot(data_samples, model_samples, dir=tmp_checkpoint_dir)
            self.save_energy_plot(
                data_samples,
                neg_samples,
                dir=tmp_checkpoint_dir,
                label=f"_neg_{self.global_step_}",
            )
            ckpt_fn = self.save_model(dir=tmp_checkpoint_dir)
            return ckpt_fn

        def _restore(self, checkpoint):
            """Restore the model"""
            self.load_model(checkpoint)

    return EnergyTrainer
