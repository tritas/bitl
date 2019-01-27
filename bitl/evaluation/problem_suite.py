# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
from numpy import array, float64, zeros


class ProblemSuite:
    """ Defines instances of problems to benchmark bandits
    in different scenarios. """

    def __init__(self):
        pass


class LinearSuite(ProblemSuite):
    """ Suite of problems for linear bandits """

    def __init__(self):
        super(LinearSuite).__init__()


def bernouilli_hundred():
    """ Problem for 100 arms of Bernouilli rewards of rare high rewards

    Returns
    -------
    R: array of shape [100]
        Vector of arms' mean rewards

    """
    R = zeros(100, dtype=float64)
    R[:80] = 0.3
    R[80:90] = 0.5
    R[90:95] = 0.6
    R[95:98] = 0.7
    R[98:99] = 0.8
    R[-1] = 0.95
    return R


def bernouilli_thousand():
    """ Problem for 1000 arms of Bernouilli rewards with few high rewards

    Returns
    -------
    R: array of shape [100]
        Vector of arms' mean rewards

    """

    R = zeros(1000, dtype=float64)
    R[:900] = 0.3
    R[900:950] = 0.5
    R[950:980] = 0.6
    R[980:999] = 0.8
    R[-1] = 0.95
    return R


def bernouilli_big_gap():
    """ Easy problem """
    return array([0.5, 0.9])


def bernouilli_medium_gap():
    """ Less easy problem """
    return array([0.5, 0.7])


def bernouilli_small_gap():
    """ The smaller gap, the harder the problem.
    Will be averaging over multiple runs. """
    return array([0.5, 0.55])


class BernouilliArms(ProblemSuite):
    """ Suite of problems for arm rewards distributed as bernouilli r.v.s """
    problems = [
        bernouilli_small_gap(),
        bernouilli_medium_gap(),
        bernouilli_big_gap(),
        bernouilli_hundred()
    ]


class NonStationaryProblems(ProblemSuite):
    """ Suite of problems for non-stationary bandits """
    pass
