# -*- coding: utf-8 -*-

import numpy as np
from bitl.datasets.synthetic import stochastic_batch


def test_bernouilli():
    """
    This is a trivial test just to get things started.
    We should check that results are well-formed in expectation.
    """
    horizon = 100
    means = np.array([0.55, 0.60])
    signal = stochastic_batch(means, horizon)
    assert signal.shape == (horizon, len(means))
