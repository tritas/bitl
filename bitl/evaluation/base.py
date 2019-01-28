# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause

from time import time

import numpy as np
from ..evaluation.loops import base_kernel
from ..policies.base import Policy
from ..policies.base import policy_name
from ..utils.export import render_simple
from joblib import Parallel, delayed


def _run(eval_kernel, policy, signal, rewards, horizon, **side_info):
    policy.initialize()
    start = time()
    reward_arr, regret_arr = eval_kernel(
        policy, signal, rewards, horizon, **side_info
    )
    runtime = time() - start
    return reward_arr, regret_arr, runtime


def _replay(
    eval_kernel,
    policy,
    signal,
    rewards,
    horizon=None,
    n_runs=1,
    n_jobs=1,
    verbose=0,
    **side_info
):
    """Replay a policy on an environemnt for a number of runs and with a certain number of jobs.

    :param policy: The policy to replay
    :return:
    A metrics object containing information about the run
    """
    """
    if 'time_lapses' in kwargs:
        l = skwargs['time_lapses']
        print(l.shape[0] - horizon)
    else: # Case of linear rewards
        l = np.zeros(horizon, dtype=np.int32)
    """
    horizon = horizon or len(signal)
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_run)(eval_kernel, policy, signal, rewards, horizon)
        for _ in range(n_runs)
    )
    reward_tup, regret_tup, runtimes_tup = zip(*results)
    regret = np.stack(regret_tup)
    reward = np.stack(reward_tup)
    runtimes = np.stack(runtimes_tup)

    metrics = {
        "total_runtime": runtimes.sum(),
        "mean_reward": reward.mean(axis=0),
        "mean_regret": regret.mean(axis=0),
        "cumulative_regret": regret.mean(axis=0).cumsum(),
    }
    return metrics


class PolicyEvaluation(object):
    """ The main evaluation object. """

    metrics = dict()

    def __init__(
        self,
        rewards=None,
        signal=None,
        evaluation_kernel=None,
        n_runs=100,
        horizon=10000,
        random_seed=1,
        n_jobs=1,
        verbose=0,
    ):
        """
        Parameters
        ----------
        rewards : array of shape [n_users, n_items]
            The reward matrix.

        signal : array of shape [n_samples, n_objects]
            The stream array

        evaluation_kernel : callable (optional)
            The function that evaluates an algorithm over a pass of the stream.
            If None, use a base kernel for 1d array of bernouilli-distributed
            rewards

        n_runs: int, optional (default=100)
            The number of passes to do over the stream.
            Final metrics should be averaged over all runs.

        horizon : int, optional (default=10000)
            The number of samples to consider for each run.

        random_seed : int, optional (default=1)
            The seed used by the random number generator.

        n_jobs : int, optional (default=1)
            The number of cores to use to run the evaluations.
            If n_jobs=-1, all cores will be used.

        verbose : int, optional (default=0)
            The verbosity level of the evaluation.
        """
        # side_information = None,results_path=None,
        # self.policy = policy
        self.horizon = horizon
        self.n_runs = n_runs
        self.n_jobs = n_jobs
        self.signal = signal
        self.rewards = rewards
        self.verbosity = verbose

        if evaluation_kernel is None:
            self.eval_kernel = base_kernel
        else:
            self.eval_kernel = evaluation_kernel

        self.dry_runtime = None
        self.dry_run()

        np.random.seed(random_seed)

    def set_evaluation_kernel(self, callable_loop):
        """ Define the evaluation loop to be used for evaluation.

        Parameters
        ----------
        callable_loop: function
            A callable simulating the environment for one run
            (called multiple times)
        """
        self.eval_kernel = callable_loop

    def dry_run(self):
        """ Measuring policy evaluator overhead """
        self.replay(Policy())
        self.dry_runtime = self.metrics["DummyPolicy"]["total_runtime"]
        del self.metrics["DummyPolicy"]

    def replay(self, policy, **side_info):
        """
        The main evaluation code is called outside of the object so that
        parallel calls' memory is well separated.
        :param policy:
        """
        metrics = _replay(
            self.eval_kernel,
            policy,
            self.signal,
            self.rewards,
            self.horizon,
            self.n_runs,
            self.n_jobs,
            self.verbosity,
            **side_info
        )
        self.metrics[policy_name(policy)] = metrics

    # TODO(aris): dissociate the output from the main evaluation code
    def print_metrics(self):
        """ Print the results of the latest replay """
        for algorithm, metrics in self.metrics.items():
            print(
                "[{}] Runtime: {:.3f} s/run -"
                "Final Regret: {:.1f}".format(
                    algorithm,
                    metrics["total_runtime"],
                    metrics["cumulative_regret"][-1],
                )
            )

    # TODO(aris): get rid of that
    def output_html(self, policy):
        """ Metrics to HTML """
        algorithm = policy_name(policy)
        metrics = self.metrics[algorithm]
        metrics["final_regret"] = metrics["cumulative_regret"][-1]
        html = render_simple(metrics, "synthetic.html")
        return html

    # TODO(aris): get rid of that
    def output_tex(self, policy):
        """ Metrics to LaTeX"""
        raise NotImplementedError
