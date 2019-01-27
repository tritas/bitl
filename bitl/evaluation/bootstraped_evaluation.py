# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
from time import time

import numpy as np
from joblib import Parallel, delayed

from .evaluation import PolicyEvaluation
from ..utils.imports import policy_name


def _bootstraped_replay(policy, signal, evaluation_kernel, bootstrap, jitter,
                        n_jobs, verbosity):
    """Evaluate a policy on a log using a bootstraped approach.

    Parameters
    ----------
    policy : Policy to evaluate

    Returns
    -------
    mean payoff resulting from the evaluation
    """

    horizon = len(signal)
    start = time()

    if n_jobs == 1:
        n_valid = np.zeros(bootstrap, dtype=np.int32)
        payoff = np.zeros(bootstrap, dtype=np.float64)
        for b in range(bootstrap):
            n_valid[b], payoff[b] = \
                evaluation_kernel(policy, signal, horizon, jitter)
    else:
        results = Parallel(n_jobs=n_jobs, verbose=verbosity)(
            delayed(evaluation_kernel)(policy, signal, horizon, jitter)
            for _ in range(bootstrap))
        valid_seq, payoff_seq = zip(*results)
        payoff = np.stack(payoff_seq)
        n_valid = np.stack(valid_seq)

    run_t = time() - start

    metrics = {
        'total_runtime': run_t,
        'bagged_mean_payoff': np.mean(np.divide(payoff, n_valid)),
        'summed_mean_payoff': np.sum(payoff) / np.sum(n_valid)
    }
    return metrics


class BootstrapedEvaluation(PolicyEvaluation):
    """ Evaluate a policy on a stream using bootstraped replay estimation.

    Reference
    ---------
    Nicol, O., Mary, J., & Preux, P. (2014, June).
    Improving offline evaluation of contextual bandit algorithms
    via bootstrapping techniques.
    In Proceedings of the 31th International Conference on Machine Learning
     (ICML-2014)

    """

    def __init__(self, evaluation_kernel=None,
                 n_runs=100, horizon=10000, random_seed=1, n_jobs=1,
                 verbose=0, bootstrap=100, jitter=0.5):
        self.jitter = jitter
        self.bootstrap = bootstrap
        super(BootstrapedEvaluation, self).__init__(
            evaluation_kernel=evaluation_kernel, n_runs=n_runs,
            horizon=horizon, random_seed=random_seed, n_jobs=n_jobs,
            verbose=verbose)

    def replay(self, policy, **side_info):
        """
        Call the main evaluation loop

        Parameters
        ----------
        policy : policy class
             The class that will be evaluated
        """
        metrics = _bootstraped_replay(policy,
                                      self.signal,
                                      self.eval_kernel,
                                      self.bootstrap,
                                      self.jitter,
                                      self.n_jobs,
                                      self.verbosity)

        self.metrics[policy_name(policy)] = metrics

    def replay_all(self, policies):
        """ Runs simulations in parallel """
        t0 = time()
        for p in policies:
            self.replay(p)
        print("Comparison took {:.3f} seconds".format(time() - t0))

    def output_html(self, policy):
        """
        Format an evalutation output to HTML
        :param policy:
        :return: HTML containing the results
        """
        algorithm_name = policy_name(policy)
        metrics = self.metrics[algorithm_name]
        html = '<br/><strong>{}</strong><br/> Mean payoff: bagged = {},  summed = {}<br/> Runtime: {}'.format(algorithm_name, metrics['bagged_mean_payoff'], metrics['summed_mean_payoff'], metrics['total_runtime'])
        return html

    def output_tex(self):
        pass