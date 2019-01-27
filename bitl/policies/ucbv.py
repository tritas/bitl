# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
import numpy as np


class UCBV:
    """
    Reference
    ---------

    Bubeck, S., Stoltz, G., Szepesvári, C., & Munos, R. (2009).
    Online optimization in X-armed bandits.
    In Advances in Neural Information Processing Systems (pp. 201-208).
    """

    def __init__(self, *args, **kwargs):
        self.alpha = kwargs["alpha"]
        self.horizon = kwargs["horizon"]
        self.beta = kwargs["beta"]
        self.mu_star = kwargs["mu_star"]

        if self.beta < 1 and self.mu_star < 1:
            self.n_arms = int(round(self.horizon ** (self.beta / 2)))
        else:
            self.n_arms = int(round(self.horizon ** (self.beta / (self.beta + 1))))

        self.mu_hat = None
        self.v = None
        self.n_draws = None
        self.payoffs = None
        self.t = 0

    def initialize(self):
        """ Initialize the algorithm """
        self.payoffs = np.zeros((self.n_arms, self.horizon), dtype=np.float64)
        self.mu_hat = np.zeros(self.n_arms, dtype=np.float64)
        self.v = np.zeros(self.n_arms, dtype=np.float64)
        self.n_draws = np.zeros(self.n_arms, dtype=np.int32)
        self.t = 0

    def select_arm(self):
        for i in range(self.n_arms):
            if self.n_draws[i] == 0:
                return i

        expl_t = np.log(10 * np.log(self.t))  # + np.log(t)/2
        self.mu_hat = self.payoffs / self.n_draws
        self.v = (
            np.power(self.payoffs, 2).sum(axis=1)
            - np.power(self.payoffs, 2) / self.n_draws
        ) / (self.n_draws - 1)
        expl_bonus = np.sqrt(2 * self.v * expl_t / self.n_draws) + (
            3 * expl_t / self.n_draws
        )
        bound = self.mu_hat + expl_bonus

        return bound.argmax()

    def update(self, a, x):
        self.payoffs[a, self.t] += x
        self.n_draws[a] += 1
        self.t += 1
