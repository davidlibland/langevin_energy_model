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

n_dim = 1

X = pd.read_csv("1d_mixture.csv", header=None, index_col=None)
X

plt.hist(X.values.flatten(), alpha=0.5)
plt.show()

do_search= "bohb"# "halving"
warm = False

clf = EnergyModel(
    n_dim,
    batch_size=5000,
    num_layers=3,
    num_units=12,
    weight_decay=3e-3,
    max_iter=1000,
    num_mc_steps=100,
    replay_prob=1,
    sampler="langevin",
    lr=0.01,
#     sampler_lr=1e-2,
    prior_scale=10,
    adversary_weight=0.0,
    num_sample_mc_steps=1000,
    sampler_beta_min=0.02,
    sampler_beta_target=10,
    max_replay=1
)
max_resources = 1000
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
    distributions.add_hyperparameter(CSH.UniformFloatHyperparameter("sampler_beta_target", 1, 1e2, default_value=1, log=True))
    distributions.add_hyperparameter(CSH.UniformFloatHyperparameter("sampler_beta_min", 1e-3, 1-1e-2, default_value=.1, log=True))
#     distributions.add_hyperparameter(CSH.UniformFloatHyperparameter("replay_prob", 0, 1, default_value=0.75))
#     distributions.add_hyperparameter(CSH.UniformFloatHyperparameter("adversary_weight", 0, .1, default_value=0.05))
    distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("num_units", 4, 32, default_value=16))
    distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("num_layers", 2, 4, default_value=3))
#     distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("max_replay", 2, 20, default_value=10))

    clf_cv = HpBandSterSearchCV(
        clf, 
        distributions, 
        random_state=0, 
        n_jobs=6,
        resource_name='max_iter', # can be either 'n_samples' or a string corresponding to an estimator attribute, eg. 'n_estimators' for an ensemble
        resource_type=int, # if specified, the resource value will be cast to that type before being passed to the estimator, otherwise it will be derived automatically
        min_budget=100,
        max_budget=max_resources,
        refit=True,
        optimizer="bohb",
    )
    search = clf_cv.fit(X, y=None)
    clf = clf_cv.best_estimator_
else:
    clf.fit(X.values)

search.best_params_

clf.num_sample_mc_steps = 1000
samples = clf.sample(1000)

fig, ax = plt.subplots()
ax.hist([X.values.flatten(), samples.flatten()], alpha=0.5, density=True)
plt.show()

fig.savefig("mixture.png")

cols = ["energy_diff", "energy_coef", "data_erf", "adversary_cost"]
logs = pd.DataFrame({k:v for k,v in clf.logger_.full_logs.items() if k in cols})

plt.plot(clf.logger_.full_logs["loss_ais"])

logs["energy_coef"].apply(np.log).plot()

logs[[c for c in cols if c != "energy_diff"]].plot()

import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis

result = search._res

# +
# get all executed runs
all_runs = result.get_all_runs()

# get the 'dict' that translates config ids to the actual configurations
id2conf = result.get_id2config_mapping()

lcs = result.get_learning_curves()

hpvis.interactive_HBS_plot(lcs)
# -

result.get_all_runs()[0].info['test_score_mean'], result.get_all_runs()[0].loss

# +
# Here is how you get he incumbent (best configuration)
inc_id = result.get_incumbent_id()

# let's grab the run on the highest budget
inc_runs = result.get_runs_by_id(inc_id)
inc_run = inc_runs[-1]


# We have access to all information: the config, the loss observed during
#optimization, and all the additional information
inc_loss = inc_run.loss
inc_config = id2conf[inc_id]['config']
inc_test_loss = inc_run.info['test_score_mean']

print('Best found configuration:')
print(inc_config)
print('It achieved accuracies of %f (validation) and %f (test).'%(1-inc_loss, inc_test_loss))


# Let's plot the observed losses grouped by budget,
hpvis.losses_over_time(all_runs)

# the number of concurent runs,
hpvis.concurrent_runs_over_time(all_runs)

# and the number of finished runs.
hpvis.finished_runs_over_time(all_runs)

# This one visualizes the spearman rank correlation coefficients of the losses
# between different budgets.
hpvis.correlation_across_budgets(result)

# For model based optimizers, one might wonder how much the model actually helped.
# The next plot compares the performance of configs picked by the model vs. random ones
hpvis.performance_histogram_model_vs_random(all_runs, id2conf)

plt.show()

# +
runs = all_runs
get_loss_from_run_fn = lambda r: r.loss
budgets = set([r.budget for r in runs])

data = {}
for b in budgets:
    data[b] = []

for r in runs:
    if r.loss is None:
        continue
    b = r.budget
    t = r.time_stamps['finished']
    l = get_loss_from_run_fn(r)
    data[b].append((t,l))

for b in budgets:
    data[b].sort()
# -


