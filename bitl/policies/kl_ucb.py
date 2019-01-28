# -*- coding: utf-8 -*-
import numpy as np

EPSILON = 1e-6


def klucb(action_draws, action_mean, total_draws):
    """ Compute the KL-divergence for an arm index """
    if not action_draws:
        return np.infty

    mean = (action_mean + 1) / 2
    bonus = np.log(total_draws / action_draws + 1)
    bonus += np.log(bonus)
    bonus /= action_draws

    delta = (1 - action_mean) / 4

    while delta > EPSILON:
        if mean <= 0:
            return EPSILON
        if kinf(action_mean, mean) < bonus:
            mean += delta
        else:
            mean -= delta

        delta /= 2

    return mean


def kinf(p, q):
    """ KL value between two Bernouilli-distributed r.v.s """
    p = np.clip(p, 0, 1)
    q = np.clip(q, 0, 1)
    if p > q:
        return 0
    elif p > 0 < 1 and q > 0 < 1:
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    elif p == 0:
        return np.log(1 / (1 - q))
    elif p == 1:
        return np.log(1 / q)
    else:
        return np.infty


class klUCB:
    """
    Reference
    ---------
    Cappé, O., Garivier, A., Maillard O-A., Munos. R, Stoltz, G.
    "Kullback–leibler upper confidence bounds for optimal sequential
    allocation." The Annals of Statistics 41, no. 3 (2013): 1516-1541.
    """

    def __init__(self, n_actions):
        self.n_arms = n_actions
        self._means = None
        self._n_draws = None
        self._payoffs = None
        self._ucb = None
        self._t = 0
        self._tot_draws = 0

    def initialize(self):
        """ Initialize the algorithm """
        self._payoffs = np.zeros(self.n_arms, dtype=np.float64)
        self._means = np.zeros(self.n_arms, dtype=np.float64)
        self._n_draws = np.zeros(self.n_arms, dtype=np.int32)
        self._ucb = np.full(self.n_arms, np.infty, dtype=np.float64)
        self._t = 0
        self._tot_draws = 0

    def select_arm(self, *args):
        """ Choose the best arm."""
        for i, draws in enumerate(self._n_draws):
            if not draws:
                return i

        return self._ucb.argmax()

    def update(self, action, reward):
        """ Update stats """
        self._tot_draws += 1
        self._payoffs[action] += reward
        self._n_draws[action] += 1
        self._means[action] = self._payoffs[action] / self._n_draws[action]
        self._ucb[action] = klucb(
            self._n_draws[action], self._means[action], self._tot_draws
        )
        self._t += 1
