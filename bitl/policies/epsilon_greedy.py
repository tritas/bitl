# -*- coding: utf-8 -*-
import numpy as np


class EpsilonGreedy:
    """ Exploring the exploration/exploitation tradeoff """

    def __init__(self, K, epsilon, annealing=False, softmax=False):
        self.t = 0
        self.n_arms = K
        self.eps = epsilon
        self.annealing = annealing
        self.is_softmax = softmax

        self.means = None
        self.n_draws = None
        self.payoffs = None

    def initialize(self):
        """ Initialize the algorithm """
        self.payoffs = np.zeros(self.n_arms, dtype=np.float64)
        self.means = np.zeros(self.n_arms, dtype=np.float64)
        self.n_draws = np.zeros(self.n_arms, dtype=np.int32)
        self.t = 0

    def select_arm(self):
        """ Choose arm:
        Pr(eps) -> exploration
        Pr(1-eps) -> exploitation """

        for i in range(self.n_arms):
            if not self.n_draws[i]:
                return i

        # Exploration
        if np.random.binomial(1, self.eps):
            if self.is_softmax:
                if self.annealing:
                    tau = 1 / np.log(self.t + 1e-07)
                else:
                    tau = 1
                probs = np.exp(tau * self.means)
                return np.divide(probs, probs.sum()).argmax()
            else:
                return np.random.randint(self.n_arms)
        # Exploitation
        else:
            return self.means.argmax()

    def update(self, arm, reward):
        """ Update stats with
        Parameters
        ----------
        arm : the chosen arm
        reward : the obtained reward
        """
        self.payoffs[arm] += reward
        self.n_draws[arm] += 1
        self.means[arm] = self.payoffs[arm] / self.n_draws[arm]
        self.t += 1
        self.eps = 1 / np.log(self.t + 1e-07)

    def __str__(self):
        return "Epsilon Greedy with epsilon={}".format(str(self.eps))
