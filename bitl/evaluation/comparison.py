# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
import sys
from time import time

from .base import PolicyEvaluation


class PolicyComparison(PolicyEvaluation):
    """
    An object to run the evaluation on multiple policies
    """

    def __init__(
        self,
        policies,
        rewards,
        signal,
        evaluation_kernel=None,
        n_runs=100,
        horizon=10000,
        random_seed=1,
        n_jobs=1,
        verbose=0,
    ):
        self.policies = policies
        super(PolicyComparison, self).__init__(
            rewards=rewards,
            signal=signal,
            evaluation_kernel=evaluation_kernel,
            n_runs=n_runs,
            horizon=horizon,
            random_seed=random_seed,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    # TODO(aris): run them in parallel
    def replay_all(self):
        """ Runs simulations. """
        t0 = time()
        for policy in self.policies:
            self.replay(policy)
        sys.stdout.write("Comparison took {:.3f} seconds\n".format(time() - t0))

    def output_html(self, policy):
        """ TODO: Output metrics as HTML """
        raise NotImplementedError

    def output_tex(self, policy):
        """ TODO: Output metrics as LaTeX """
        raise NotImplementedError
