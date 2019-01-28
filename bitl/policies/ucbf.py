# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
import numpy as np


class UCBF:
    """
    Reference
    ---------

    Bubeck, S., Stoltz, G., Szepesvári, C., & Munos, R. (2009).
    Online optimization in X-armed bandits.
    In Advances in Neural Information Processing Systems (pp. 201-208).
    """

    def __init__(self, *args, **kwargs):
        if "K" not in kwargs:
            self.horizon = kwargs["horizon"]
            self.beta = kwargs["beta"]
            self.mu_star = kwargs["mu_star"]
            if self.beta < 1 and self.mu_star < 1:
                self.n_arms = round(self.horizon ** (self.beta / 2))
            else:
                self.n_arms = round(
                    self.horizon ** (self.beta / (self.beta + 1))
                )
        else:
            self.n_arms = kwargs["K"]

        self.means = None
        self.var_est = None
        self.M2 = None
        self.n_draws = None
        self.payoffs = None

    # Initialize the algorithm
    def initialize(self):
        self.payoffs = np.zeros(self.n_arms, dtype=np.float64)
        self.means = np.zeros(self.n_arms, dtype=np.float64)
        self.var_est = np.zeros(self.n_arms, dtype=np.float64)
        self.n_draws = np.zeros(self.n_arms, dtype=np.int32)
        self.M2 = np.zeros(self.n_arms, dtype=np.float64)
        self.t = 0

    def select_arm(self):
        for i in range(self.n_arms):
            if self.n_draws[i] == 0:
                return i

        e_t = np.log(10 * np.log(self.t + 10))
        bound = (
            self.means
            + np.sqrt(2 * self.var_est * e_t / self.n_draws)
            + (3 * e_t / self.n_draws)
        )
        return bound.argmax()

    def update(self, a, X):
        """ Online update of the mean and the variance using moments

        Reference
        ---------
        B. P. Welford (1962). "Note on a method for calculating corrected
        sums of squares and products". Technometrics 4(3):419–420.
        """
        self.payoffs[a] += X
        self.n_draws[a] += 1

        delta = X - self.means[a]
        self.means[a] += delta / self.n_draws[a]
        self.M2[a] += delta * (X - self.means[a])

        if self.n_draws[a] > 2:
            self.var_est[a] = self.M2[a] / (self.n_draws[a] - 1)
        else:
            self.var_est[a] = np.nan

        self.t += 1
