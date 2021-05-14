"""
An API for the model which conforms to sklearn's interface.
In particular, it has a `fit` and `sample` method, mimicking the Gaussian Mixture Model.
"""
import itertools
import time
from collections import deque
import random
import math

import scipy.spatial.distance
import numpy as np
import torch
from torch import optim
import torch.utils.data as data
import sklearn.ensemble as sk_ensemble

import energy_model.model
import energy_model.utils
import energy_model.utils.beta_schedules
import energy_model.mcmc
import energy_model.mcmc.langevin
import energy_model.mcmc.tempered_transitions
import energy_model.mcmc.mala
from energy_model.hparam_sweep import Logger
from energy_model.utils.ais import AISLoss


def e_coef(data_samples, model_samples):
    """Computes the energy coefficient"""
    d1 = scipy.spatial.distance.cdist(data_samples, model_samples).mean()
    d2 = scipy.spatial.distance.cdist(data_samples, data_samples).mean()
    d3 = scipy.spatial.distance.cdist(model_samples, model_samples).mean()
    return math.sqrt(max(0, (2 * d1 - d2 - d3) / max(2 * d1, np.finfo(np.float).eps)))


class EnergyModel:
    def __init__(self, num_inputs, **kwargs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.set_params(num_inputs=num_inputs, **kwargs)
        self.verbose = True

    def fit(self, X, y=None):
        """
        Fits the energy model to X.

        Args:
            X: The dataset, of shape (num_samples, num_inputs)
        """
        del y  # unused

        epoch_training_time = 0
        epoch_metrics_time = 0

        dataloader = data.DataLoader(
            dataset=np.array(X, dtype=np.float32),
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
        )
        adversary = None

        for epoch in range(self.epoch, self.max_iter):
            for i_batch, sample_batched in enumerate(dataloader):
                batch_start_time = time.time()
                data_sample = sample_batched.to(self.device)

                # Get model samples, either from replay buffer or noise.
                if len(self.model_samples_) > self.max_replay:
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
                    mc_dynamics=self.sampler_,
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
                data_energy = self.net_(data_sample)
                data_energy_mean = data_energy.mean()
                model_energy = self.net_(model_sample)
                model_energy_mean = model_energy.mean()
                self.logger_(energy_coef=e_coef(data_sample, model_sample))

                # Estimate the odds of the data's energy based on a normal fitted to
                # model samples:
                data_erf = torch.erf(
                    (data_energy_mean - model_energy_mean) / model_energy.std()
                )

                objective = data_energy.sum() - model_energy.sum()

                # Potentially add the adversary:
                if adversary:
                    y = adversary.predict_proba(
                        model_sample.detach().cpu().numpy()
                    ).astype(np.float32)
                    weight = (y @ adversary.classes_).reshape([-1, 1])
                    self.logger_(adversary_cost=weight.mean())
                    adversary_energy = -(model_energy * torch.tensor(weight)).sum()
                    objective = (
                        self.adversary_weight * adversary_energy
                        + (1 - self.adversary_weight) * objective
                    )
                else:
                    self.logger_(adversary_cost=1.0)

                objective.backward()
                torch.nn.utils.clip_grad.clip_grad_value_(self.net_.parameters(), 1e2)
                self.optimizer.step()

                if self.adversary_weight:
                    adversary = sk_ensemble.RandomForestClassifier(class_weight="balanced")
                    X = np.concatenate(
                        [data_sample.detach().cpu().numpy()]
                        + [
                            m_sample.detach().cpu().numpy()
                            for m_sample in self.model_samples_
                        ],
                        axis=0,
                    )
                    y = np.concatenate(
                        [-np.ones(shape=data_sample.shape[:1])]
                        + [
                            np.ones(shape=m_sample.shape[:1])
                            for m_sample in self.model_samples_
                        ],
                        axis=0,
                    )
                    adversary.fit(X, y)

                batch_training_time = time.time() - batch_start_time
                epoch_training_time += batch_training_time
                self.logger_(energy_diff=float(objective))
                self.logger_(data_erf=float(data_erf))

                tr_metrics_start_time = time.time()
                if self.use_ais and i_batch == 0 and epoch % 10 == 0:
                    self.ais_loss(net=self.net_, data_sample=data_sample, global_step=epoch)
                tr_metrics_time = time.time() - tr_metrics_start_time
                epoch_metrics_time += tr_metrics_time
                if self.verbose:
                    print(
                        f"on epoch {epoch}, batch {i_batch}, data erf: {data_erf}, objective: {objective}"
                    )
                    print(f"model energy: {model_energy_mean} +- {model_energy.std()}")
                    print(f"data energy: {data_energy_mean}")
                    print(
                        f"training time: {batch_training_time:0.3f}s, metrics time: {tr_metrics_time:0.3f}s"
                    )
            means = self.logger_.means()
            if self.verbose:
                print(f"on epoch {epoch}")
                for k, v in means.items():
                    print(f"{k}: {v}")
            self.logger_.flush()
            self.epoch = epoch

    def sample(self, n_samples):
        """Return samples"""
        sample_list = []
        for x in itertools.cycle(reversed(self.model_samples_)):
            model_samples = (
                self.net_.sample_fantasy(
                    x=x,
                    num_mc_steps=self.num_sample_mc_steps,
                    beta=self.sample_beta,
                    mc_dynamics=self.sampler_,
                )
                .detach()
                .cpu()
                .numpy()
            )
            sample_list.append(model_samples)
            if sum(x_.shape[0] for x_ in sample_list) >= n_samples:
                break
        return np.concatenate(sample_list, axis=0)[:n_samples, ...]

    def score(self, X, y=None):
        """Returns the (negative) energy coefficient"""
        del y  # unused
        n_samples = X.shape[0]

        if self.use_ais:
            data_sample = torch.tensor(np.array(X).astype(np.float32))
            return -self.ais_loss(net=self.net_, data_sample=data_sample, global_step=self.epoch)
        # Otherwise, use the energy coefficient
        model_samples = self.sample(n_samples)
        return -np.log(e_coef(X, model_samples))

    def get_params(self, deep=True):
        """
        Returns the hyperparmeters. Conforms to sklearn' api
        (allowing this class to be used with hparam tuning strategies)
        """
        return self.kwargs

    def set_params(self, **kwargs):
        """
        Set the parameters. Conforms to sklearn' api
        (allowing this class to be used with hparam tuning strategies)

        Args:
            **kwargs: Parameters to set
        """
        if not hasattr(self, "kwargs"):
            self.kwargs = {}
        self.kwargs = {**self.kwargs, **kwargs}
        kwargs = self.kwargs
        self.use_ais = kwargs.get("use_ais", False)
        self.adversary_weight = kwargs.get("adversary_weight")
        self.batch_size = kwargs.get("batch_size", 1024)
        self.weight_decay = kwargs.get("weight_decay", 1e-3)
        self.lr = kwargs.get("lr", 1e-2)
        self.max_iter = kwargs.get("max_iter", 100)
        self.num_mc_steps = kwargs.get("num_mc_steps", 10)
        self.max_replay = kwargs.get("max_replay", 10)
        self.replay_prob = kwargs.get("replay_prob", 0.9)
        sampler = kwargs.get("sampler", "mala")
        num_tempered_transitions = kwargs.get("num_tempered_transitions", 1)
        if "tempered" in sampler:
            self.num_mc_steps = num_tempered_transitions
        self.num_tempered_transitions = num_tempered_transitions
        self.num_sample_mc_steps = kwargs.get("num_sample_mc_steps", 100)
        self.sample_size = kwargs.get("sample_size", 1000)
        self.sampler_lr = kwargs.get("sampler_lr", 10 * self.lr)
        self.beta_target = kwargs.get("sampler_beta_target", 1)
        self.sample_beta = kwargs.get("sample_beta", self.beta_target)
        self.sampler_beta_min = kwargs.get("sampler_beta_min", 0.1)
        # Reweight number of steps if tempering is used:
        sampler_beta_schedule_num_steps = max(
            self.num_mc_steps // self.num_tempered_transitions, 1
        )
        sampler_beta_schedule = energy_model.utils.beta_schedules.build_schedule(
            ("geom", self.beta_target, sampler_beta_schedule_num_steps,),
            start=self.sampler_beta_min,
        )
        if not kwargs.get("warm_start") or not hasattr(self, "net_"):
            self.net_ = energy_model.model.SimpleEnergyModel(
                kwargs.get("num_inputs"),
                kwargs.get("num_layers", 3),
                kwargs.get("num_units", 16),
                kwargs.get("prior_scale", 1),
            )
            self.net_.to(self.device)
            self.optimizer = optim.Adam(
                self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            self.logger_ = Logger()
            samplers = {
                "mala": energy_model.mcmc.mala.MALASampler(
                    lr=self.sampler_lr, beta=self.beta_target, logger=self.logger_
                ),
                "langevin": energy_model.mcmc.langevin.LangevinSampler(
                    lr=self.sampler_lr, beta=self.beta_target, logger=self.logger_,
                ),
                "tempered langevin": energy_model.mcmc.tempered_transitions.TemperedTransitions(
                    mc_dynamics=energy_model.mcmc.langevin.LangevinSampler(
                        lr=self.sampler_lr, logger=self.logger_
                    ),
                    beta_schedule=sampler_beta_schedule,
                    logger=self.logger_,
                ),
                "tempered mala": energy_model.mcmc.tempered_transitions.TemperedTransitions(
                    mc_dynamics=energy_model.mcmc.mala.MALASampler(
                        lr=self.sampler_lr, logger=self.logger_
                    ),
                    beta_schedule=sampler_beta_schedule,
                    logger=self.logger_,
                ),
            }
            self.model_samples_ = deque(
                [
                    self.net_.sample_from_prior(
                        self.batch_size, device=self.device
                    ).detach()
                ]
            )
            self.sampler_ = samplers.get(sampler)
            self.epoch = 0

            self.ais_loss = AISLoss(
                logger=self.logger_,
                num_chains=kwargs.get("ais_num_chains", 100),
                log_z_update_interval=kwargs.get("ais_update_interval", 32),
                max_interpolants=kwargs.get("ais_max_interpolants", 5000),
                num_interpolants=kwargs.get("ais_num_interpolants", 500),
            )
        return self
