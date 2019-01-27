# -*- coding: utf-8 -*-
# Author: Aris Tritas
# License: BSD 3-clause

import os

import numpy as np

from bitl.evaluation.comparison import PolicyComparison
from bitl.policies.kl_ucb import klUCB
from bitl.policies.random import RandomAction
from bitl.policies.thompson_sampling import ThompsonSampling
from datasets.synthetic import stochastic_batch

output_dir = os.path.expanduser("~/Desktop/")
reward_matrix = np.array([0.2, 0.8])
n_actions = len(reward_matrix)
horizon = 10000
# Pick initialized policy/policies
policies = [klUCB(K=n_actions), ThompsonSampling(K=n_actions), RandomAction(K=n_actions)]
# Generate some signal or load real-world cleaned and formatted dataset
signal = stochastic_batch(reward_matrix, horizon)
# Initialize simulation object with simul parameters
# (n_runs, results_folder, ...)
sim = PolicyComparison(signal=signal, policies=policies,
                       rewards=reward_matrix, horizon=horizon, n_runs=10, n_jobs=8)
# Run simulation and get report as plots (+ table à la BESA)
sim.replay_all()
sim.print_metrics()
