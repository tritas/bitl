# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
import numpy as np


class LinUCB(object):
    """ Linear UCB with Disjoint Linear Models
    TODO: Compute and cache the inverse of A_0 and all A_a periodically"""

    def __init__(self, K, d, alpha):

        self.alpha = alpha
        assert self.alpha > 0, "alpha should be in R+"
        self.n_arms = K
        self.n_features = d

        self.n_draws = None
        self.A = None
        self.b = None
        self.theta = None
        self.p = None

        self.initialize()

    def initialize(self):
        self.A = np.zeros(
            (self.n_arms, self.n_features, self.n_features), dtype=np.float64
        )
        self.b = np.zeros((self.n_arms, self.n_features), dtype=np.float64)
        self.theta = np.zeros((self.n_arms, self.n_features), dtype=np.float64)
        self.p = np.zeros(self.n_arms, dtype=np.float64)
        self.n_draws = np.zeros(self.n_arms, dtype=np.int32)

    def select_arm(self, X):
        # Observe features of all arms: x_t,a in R^n_f -> X_t
        for i in range(self.n_arms):
            if self.n_draws[i] == 0:
                self.A[i] = np.eye(self.n_features, dtype=np.float64)
                self.b[i] = np.zeros(self.n_features, dtype=np.float64)
                return i

            A_i_inv = np.linalg.inv(self.A[i])
            self.theta[i] = np.dot(A_i_inv, self.b[i])
            self.p[i] = np.dot(self.theta[i].T, X) + self.alpha * np.sqrt(
                np.dot(X.T, np.dot(A_i_inv, X))
            )
        # Choose best arm
        return self.p.argmax()

    def get_reward(self, arm, X):
        return np.dot(self.theta[arm], X)

    def update(self, arm, reward, X):
        self.n_draws[arm] += 1
        self.A[arm] += np.dot(X, X.T)
        self.b[arm] += reward * X
