# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
import numpy as np


class ThompsonSampling(object):
    """ Agent implementing Thompson sampling for Bernouilli-distributed r.v.s
    Reference
    ---------
    William R. Thompson.
    On the likelihood that one unknown probability exceeds another
    in view of the evidence of two samples.
    Biometrika, 25(3–4):285–294, 1933.
    """

    def __init__(self, K):
        self.n_arms = K
        self.posterior = None
        self.means = None

    def initialize(self):
        """ Reset posterior as (alpha_0, beta_0)"""
        self.posterior = np.ones((self.n_arms, 2), dtype=np.int32)
        self.means = np.zeros(self.n_arms, dtype=np.float64)

    def select_arm(self):
        """Sample from the Beta distribution to infer each arms' expected reward
        :return: action: int
        """
        self.means = np.random.beta(self.posterior[:, 1], self.posterior[:, 0])
        action = self.means.argmax()
        return action

    def update(self, action, reward):
        """ Update the posterior """
        self.posterior[action][reward] += 1
