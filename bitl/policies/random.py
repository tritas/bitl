# -*- coding: utf-8 -*-
import numpy as np

from .base import Policy


class RandomWalkDiscovery(Policy):
    """Randomly choose an action (discovery setting)"""

    def __init__(self, *args, **kwargs):
        self.seq = np.arange(kwargs["n_items"])
        self.t = 0

    def initialize(self):
        self.t = 0
        np.random.shuffle(self.seq)

    def choose(self):
        return self.seq[self.t]

    def update(self, it, rt):
        self.t += 1

    def __str__(self):
        return "RandomWalkDiscovery"


class RandomAction:
    """Randomly choose an action (bandit setting)"""

    def __init__(self, K):
        self.n_arms = K

    def initialize(self):
        pass

    def select_arm(self):
        return np.random.randint(self.n_arms)

    def update(self, it, rt):
        pass

    def __str__(self):
        return "RandomAction"
