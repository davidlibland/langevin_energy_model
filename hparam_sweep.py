import os
import random
import time
from collections import deque
from typing import Any
from typing import Callable
from typing import Tuple

import torch
from ray.tune import Trainable
import matplotlib.pyplot as plt
import torch.utils.data as data
from torch import optim

import mcmc.mala
import mcmc.langevin
import mcmc.tempered_transitions
from distributions import core
from utils.ais import AISLoss
import model

# Globals (make these configurable)

MAX_REPLAY = 30
REPLAY_PROB = .99


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

    def means(self):
        """Returns the means of the metrics"""
        return {
            k: 0 if not vs else sum(vs)/len(vs)
            for k, vs in self.logs.items()
        }


def get_energy_trainer(setup_dist: Callable[[Any], Tuple[core.Sampler, model.BaseEnergyModel]]):
    """Returns a tune trainable for the distribution and model architecture"""
    class EnergyTrainer(Trainable):
        def _setup(self, config):
            self.lr = config.get("lr", 1e-2)
            self.num_mc_steps = config.get("num_mc_steps", 1)
            self.num_sample_mc_steps = config.get("num_sample_mc_steps", 1000)
            self.sample_beta = config.get("sample_beta", 1e1)
            self.batch_size = config.get("batch_size", 1024)
            self.sample_size = config.get("sample_size", 10000)
            samplers = {
                "mala": mcmc.mala.MALASampler(lr=0.1),
                "langevin": mcmc.langevin.LangevinSampler(lr=0.1),
                "tempered mala": mcmc.tempered_transitions.SimulatedTempering(mc_dynamics=mcmc.mala.MALASampler(lr=0.1)),
            }
            self.sampler = samplers.get(config.get("sampler", "mala"))
            self.verbose = True

            self.dist, self.net_ = setup_dist(**config)
            samples = self.dist.rvs(self.sample_size)
            print(samples.shape)
            dataset = data.TensorDataset(torch.tensor(samples, dtype=torch.float))
            self.energy = lambda x: self.net_(torch.tensor(x, dtype=torch.float)).detach().numpy()

            self.optimizer_ = optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-1)

            self.dataloader = data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True
            )
            # Determine the device type:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net_.to(self.device)
            print(f"training on {len(dataset)} samples using {self.device}.")
            self.global_step_ = 0
            self.epoch_ = 0
            self.model_samples_ = None

            self.logger_ = Logger()
            self.ais_loss = AISLoss(logger=self.logger_, log_z_update_interval=9)
            self.step_callbacks = [self.ais_loss]

        def save_images(self, label=None, dir=""):
            """Saves sample images at the giving dir"""
            if label is None:
                label=self.global_step_
            fig = plt.figure()
            self.net_.eval()
            samples = self.net_.sample_fantasy(x=self.model_samples_[-1],
                                               num_mc_steps=self.num_sample_mc_steps,
                                               beta=self.sample_beta,
                                               mc_dynamics=self.sampler,
                                               num_samples=36
                                               ).detach().cpu().numpy()
            self.dist.visualize(fig, samples, self.energy)
            plot_fn = os.path.join(dir, f"samples_{label}.png")
            fig.savefig(plot_fn)
            plt.close(fig)

        def save_model(self, dir="", **kwargs):
            """Saves the model at the given dir"""
            ckpt_fn = os.path.join(dir, f"model.pkl")
            torch.save({
                    "global_step": self.global_step_,
                    "epoch": self.epoch_,
                    "model": self.net_.state_dict(),
                    "optimizer": self.optimizer_.state_dict(),
                    "model_samples": list(self.model_samples_)
                }, ckpt_fn
            )

        def load_model(self, dir=""):
            """Loads the model from the specified dir."""
            ckpt_fn = os.path.join(dir, f"model.pkl")
            checkpoint = torch.load(ckpt_fn)
            self.net_.load_state_dict(checkpoint["model"])
            self.optimizer_.load_state_dict(checkpoint["optimizer"])
            self.epoch_ = checkpoint["epoch"]
            self.global_step_ = checkpoint["global_step"]
            self.model_samples_ = deque(checkpoint["model_samples"])

        def _train(self):
            """Train the model on one epoch"""
            objectives = []
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
                        [self.net_.sample_from_prior(data_sample.shape[0], device=self.device).detach()])
                elif len(self.model_samples_) > MAX_REPLAY:
                    self.model_samples_.popleft()
                replay_sample = random.choices(
                    self.model_samples_,
                    # favor more recent samples:
                    weights=list(range(1, len(self.model_samples_) + 1))
                )[0]
                noise_sample = self.net_.sample_from_prior(replay_sample.shape[0],
                                                           device=self.device)
                mask = torch.rand(replay_sample.shape[0]) < REPLAY_PROB
                while len(mask.shape) < len(replay_sample.shape):
                    # Add extra feature-dims
                    mask.unsqueeze_(dim=-1)
                model_sample = torch.where(mask.to(self.device), replay_sample,
                                           noise_sample)

                self.net_.eval()
                model_sample = self.net_.sample_fantasy(
                    model_sample,
                    num_mc_steps=self.num_mc_steps,
                    mc_dynamics=self.sampler,
                ).detach()

                self.model_samples_.append(model_sample)

                # Forward gradient:
                self.net_.train()
                self.net_.zero_grad()
                objective = self.net_(data_sample).mean() - self.net_(model_sample).mean()
                objective.backward()
                torch.nn.utils.clip_grad.clip_grad_value_(self.net_.parameters(), 1e2)
                self.optimizer_.step()

                batch_training_time = time.time() - batch_start_time
                epoch_training_time += batch_training_time
                objectives.append(float(objective))

                tr_metrics_start_time = time.time()
                for callback in self.step_callbacks:
                    callback(net=self.net_, data_sample=data_sample,
                             model_sample=model_sample, epoch=self.epoch_,
                             global_step=self.global_step_, validation=False)
                tr_metrics_time = time.time() - tr_metrics_start_time
                epoch_metrics_time += tr_metrics_time
                if self.verbose:
                    print(f"on epoch {self.epoch_}, batch {i_batch}, objective: {objective}")
                    print(f"training time: {batch_training_time:0.3f}s, metrics time: {tr_metrics_time:0.3f}s")
            means = self.logger_.means()
            self.logger_.flush()
            return means

        def _save(self, tmp_checkpoint_dir):
            """Save the model"""
            self.save_images(dir=tmp_checkpoint_dir)
            self.save_model(dir=tmp_checkpoint_dir)
            return tmp_checkpoint_dir

        def _restore(self, checkpoint):
            """Restore the model"""
            self.load_model(checkpoint)
    return EnergyTrainer
