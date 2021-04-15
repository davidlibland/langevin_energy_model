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

from energy_model.api import EnergyModel
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import RandomizedSearchCV, HalvingRandomSearchCV
from scipy.stats import poisson, expon, beta

n_dim = 2

X = pd.read_csv("clock.csv", header=0, index_col=None)
X

plt.scatter(X["x"], X["y"], alpha=0.1)

clf = EnergyModel(
    n_dim,
    batch_size=4096,
    num_layers=3,
    num_units=32,
    weight_decay=1e-2,
    max_iter=300,
    num_mc_steps=300,
    replay_prob=0.99,
    sampler="langevin",
    lr=1e-2,
    prior_scale=10
)
distributions = dict(
    lr=expon(1e-2),
    sampler_lr=expon(1e-1),
    sampler=["mala", "langevin", "tempered mala", "tempered langevin"],
    weight_decay=expon(1e-3),
#     max_iter=poisson(30),
    replay_prob=beta(a=9, b=1),
    num_units=poisson(32),
    num_layers=poisson(3),
    max_replay=poisson(10)
)
clf_cv = HalvingRandomSearchCV(clf, distributions, random_state=0, n_jobs=3, resource="max_iter", max_resources=300)
search = clf_cv.fit(X.values)

search.best_params_

search.best_params_

clf = clf_cv.best_estimator_

samples = clf.sample(1000)

plt.scatter(X["x"], X["y"], alpha=0.1)
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1)

fig, ax = plt.subplots()
ax.scatter(X["x"], X["y"], alpha=0.1)
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.1)

fig.savefig("clock.png")

cols = ["energy_diff", "energy_coef", "data_erf"]
logs = pd.DataFrame({k:v for k,v in clf.logger_.full_logs.items() if k in cols})

logs["energy_coef"].apply(np.log).plot()

logs[[c for c in cols if c != "energy_diff"]].plot()

clf.logger_.full_logs.keys()


