# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# In this example, we train a model on the clock dataset:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pathlib

from energy_model.api import EnergyModel
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import RandomizedSearchCV, HalvingRandomSearchCV
from scipy.stats import poisson, expon, beta

from hpbandster_sklearn import HpBandSterSearchCV
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

n_dim = 2

X = pd.read_csv("clock.csv", header=0, index_col=None)
X

plt.scatter(X["x"], X["y"], alpha=0.1)

do_search=None #"bohb"
warm = True

clf = EnergyModel(
    n_dim,
    batch_size=5000,
    num_layers=3,
    num_units=32,
    weight_decay=3e-2,
    max_iter=10,
    num_mc_steps=300,
    replay_prob=0.96,
    sampler="langevin",
    lr=1e-1,
    sampler_lr=1e-1,
    prior_scale=10,
    adversary_weight=0.5
)
clf = EnergyModel(
    n_dim,
    batch_size=5000,
    num_layers=3,
    num_units=12,
    weight_decay=3e-3,
    max_iter=30000,
    num_mc_steps=100,
    replay_prob=1,
    sampler="langevin",
    lr=0.003,
    sampler_lr=1e-2,
    prior_scale=10,
    adversary_weight=0.0,
    num_sample_mc_steps=1000,
    sampler_beta_min=0.02,
    sampler_beta_target=10,
    max_replay=1
)
max_resources = 30
if do_search == "halving":
    distributions = dict(
        lr=expon(1e-2),
        sampler_lr=expon(1e-1),
        sampler=["mala", "langevin", "tempered mala", "tempered langevin"],
        weight_decay=expon(1e-3),
    #     max_iter=poisson(30),
        replay_prob=beta(a=9, b=1),
        adversary_weight=beta(a=1, b=1),
        num_units=poisson(32),
        num_layers=poisson(3),
        max_replay=poisson(10),
    )
    clf_cv = HalvingRandomSearchCV(
        clf, 
        distributions, 
        random_state=0, 
        n_jobs=5, 
        resource="max_iter", 
        max_resources=max_resources
    )
    search = clf_cv.fit(X.values)
    clf = clf_cv.best_estimator_
elif do_search == "bohb":
    distributions = CS.ConfigurationSpace(seed=42)
    distributions.add_hyperparameter(CSH.UniformFloatHyperparameter("lr", 1e-4, 3e-1, log=True, default_value=6e-3))
    distributions.add_hyperparameter(CSH.UniformFloatHyperparameter("sampler_lr", 1e-4, 3e-1, log=True, default_value=1e-3))
    distributions.add_hyperparameter(CSH.CategoricalHyperparameter("sampler", choices=["mala", "langevin", "tempered mala", "tempered langevin"]))
    distributions.add_hyperparameter(CSH.UniformFloatHyperparameter("weight_decay", 1e-4, 3e-1, log=True, default_value=5e-2))
    distributions.add_hyperparameter(CSH.UniformFloatHyperparameter("prior_scale", 1e-1, 10, log=True, default_value=1.12))
    distributions.add_hyperparameter(CSH.UniformFloatHyperparameter("beta_target", 1, 1e2, default_value=1, log=True))
    distributions.add_hyperparameter(CSH.UniformFloatHyperparameter("replay_prob", 0, 1, default_value=0.75))
    distributions.add_hyperparameter(CSH.UniformFloatHyperparameter("adversary_weight", 0, 1, default_value=0.6))
    distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("num_units", 4, 32, default_value=16))
    distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("num_layers", 2, 4, default_value=3))
    distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("max_replay", 2, 20, default_value=10))
    
    results = pathlib.Path("bohb_results.pkl")
#     if warm and results.exists():
#     Using a warm start with hpbandster-sklearn fails: it tries to find past runs.
#         with results.open("rb") as fh:
#             res = pickle.load(fh)

    clf_cv = HpBandSterSearchCV(
        clf, 
        distributions, 
        random_state=0, 
        n_jobs=6,
        resource_name='max_iter', # can be either 'n_samples' or a string corresponding to an estimator attribute, eg. 'n_estimators' for an ensemble
        resource_type=int, # if specified, the resource value will be cast to that type before being passed to the estimator, otherwise it will be derived automatically
        min_budget=1,
        max_budget=max_resources,
        refit=True,
        optimizer="bohb",
    )
    search = clf_cv.fit(X, y=None)
    clf = clf_cv.best_estimator_
    with results.open("wb") as fh:
        pickle.dump(search._res, fh)
else:
    clf.fit(X.values)

search.best_params_

samples = clf.sample(1000)

fig, ax = plt.subplots()
ax.scatter(X["x"], X["y"], alpha=0.1)
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.1)

fig.savefig("clock.png")

cols = ["energy_diff", "energy_coef", "data_erf", "adversary_cost"]
logs = pd.DataFrame({k:v for k,v in clf.logger_.full_logs.items() if k in cols})

logs["energy_coef"].plot()

logs["energy_coef"].apply(np.log).plot()

logs[[c for c in cols if c != "energy_diff"]].plot()


