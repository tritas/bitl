# -*- coding: utf-8 -*-
# Author: Aris Tritas
# License: BSD 3-clause
import numpy as np
from .base import Policy


class Oracle(Policy):
    """ Oracle agent for known means """

    def __init__(self, means):
        self.opt_arms = means.argsort()[::-1]
        self.t = 0

    def initialize(self):
        """ Reset """
        self.t = 0

    def choose(self):
        return self.opt_arms[self.t]

    def update(self, it, rt):
        self.t += 1

    def __str__(self):
        return "Oracle"


class SpectralOracle(Policy):
    """ Agent implementing the oracle on the graph """

    def __init__(self, *args, **kwargs):
        self.name = kwargs["alg"]
        _, W = kwargs["eig"]
        self.optimal_arms = np.dot(W, kwargs["means"]).argsort()[::-1]
        self.t = 0

    def initialize(self):
        """ Reset """
        self.t = 0

    def choose(self, *args, **kwargs):
        """ Return optimal arm """
        return self.optimal_arms[self.t]

    def update(self, it, rt):
        """ Upd. """
        self.t += 1

    def __str__(self):
        return "{}SpectralOracle".format(self.name)
