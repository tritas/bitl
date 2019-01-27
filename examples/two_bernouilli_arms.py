# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
import numpy as np
from bitl.evaluation.comparison import PolicyComparison
from bitl.policies.kl_ucb import klUCB
from bitl.policies.thompson_sampling import ThompsonSampling
from bitl.datasets.synthetic import stochastic_batch

# --- Global parameters
horizon = 20000
n_runs = 100
n_jobs = 4
n_actions = 2
# Fix seed
np.random.seed(1337)

# Pick policies and initialize them.
policies = [
    ThompsonSampling(K=n_actions),
    klUCB(K=n_actions)
]

# --- First problem
means = np.array([0.55, 0.60])
# Generate some signal
stream = stochastic_batch(means, horizon)
# Initialize simulation object with simul parameters
# (n_runs, results_folder, ...)
evaluator = PolicyComparison(
    policies, means, stream,
    horizon=horizon, n_runs=n_runs, n_jobs=n_jobs)
# Evaluate it and show print results
evaluator.replay_all()
evaluator.print_metrics()

# --- Second problem
means = np.array([0.2, 0.8])
stream = stochastic_batch(means, horizon)

evaluator = PolicyComparison(
    policies, means, stream,
    horizon=horizon, n_runs=n_runs, n_jobs=n_jobs)
evaluator.replay_all()
evaluator.print_metrics()
# Get report as plots & HTML table
