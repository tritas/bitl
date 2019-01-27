# -*- coding: utf-8 -*-
import os

import numpy as np

from bitl.evaluation.comparison import PolicyComparison
from bitl.policies.kl_ucb import klUCB
from bitl.policies.random import RandomAction
from bitl.policies.thompson_sampling import ThompsonSampling
from bitl.policies.ucbf import UCBF
from bitl.utils.plot import multiplot
from bitl.datasets.synthetic import stochastic_batch

reward_matrix = np.zeros(100, dtype=np.float64)
reward_matrix[:80] = 0.3
reward_matrix[80:90] = 0.5
reward_matrix[90:95] = 0.6
reward_matrix[95:98] = 0.7
reward_matrix[98:99] = 0.8
reward_matrix[99] = 0.95

n_actions = len(reward_matrix)
horizon = 50000

# Pick initialized policy/policies
policies_lst = [
    klUCB(K=n_actions),
    ThompsonSampling(K=n_actions),
    RandomAction(K=n_actions),
    UCBF(K=n_actions)
]
# Generate some signal or load real-world cleaned and formatted dataset
signal = stochastic_batch(reward_matrix, horizon)
# Initialize simulation object with simul parameters
# (n_runs, results_folder, ...)
sim = PolicyComparison(policies=policies_lst,
                       signal=signal,
                       rewards=reward_matrix,
                       horizon=horizon,
                       n_jobs=-1,
                       n_runs=400,
                       verbose=1)
# Run simulation and get report as plots (+ table à la BESA)
sim.replay_all()
sim.print_metrics()

# Filter cumulated regret
results_tupls = [(alg, metrics['cumulated_regret'])
                 for (alg, metrics) in sim.metrics.items()]
# Choose file location to save plot
output_dir = os.path.expanduser("~/Desktop/bitl/")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
# Actually plot the results
multiplot(results_tupls, horizon,
          "Cumulated Regret evolution",
          os.path.join(output_dir, "many_arms_comparison.png"),
          ("Time", "Regret"))
