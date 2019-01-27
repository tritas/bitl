# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause

import numpy as np
from scipy.stats import kstest

from bitl.datasets.synthetic import gaussian_mixture_matrix
from bitl.datasets.synthetic import latent_linear_bandits
from bitl.datasets.synthetic import linear_batch
from bitl.datasets.synthetic import linear_stream
from bitl.datasets.synthetic import two_item_groups
from bitl.utils.math import sparsify


def test_two_groups():
    means = two_item_groups(10)
    print(means)


def test_latent_space_items():
    U, V, X = latent_linear_bandits(100, 100, 5)
    ks_stat, ks_pvalue = kstest(X.ravel(), cdf='norm')
    assert ks_pvalue < 1e-3


def test_latent_space_stream():
    stream, U, V = linear_stream(11, 13, 15, 17, 0.9)
    assert stream.shape == (17, 2)


def test_factorized_matrix():
    """ Generate a reward matrix as Q + U.V^T
    (U: user matrix, V: item matrix, Q: bias)
    where 10 users on a latent space of 4 dimensions. """
    # -- User matrix
    user_matrix = gaussian_mixture_matrix(
        (10, 4),
        [(0.2, [(0, 1), (0, 0.1)]),
         (0.8, [(0, 1 / 2), (0, 1 / 5)])])
    # -- Item matrix
    item_matrix = gaussian_mixture_matrix(
        (10, 4),
        [(1, [(0, 1), (0, 0.1)])])
    item_matrix = sparsify(item_matrix, 2)
    # -- Reward matrix
    rewards = gaussian_mixture_matrix(
        (10, 10),
        [(0.8, [(0.4, 0.01)]),
         (0.2, [(0.6, 0.01)])]) + np.dot(user_matrix, item_matrix.T)
    print(rewards)


def test_linear_batch():
    params = {
        'n_samples': 100,
        'dims': (20, 10, 10),
        'noise_var': 0.1,
        'mixtures': [(12, [(0, 1), (0, 0.1)]),
                     (4, [(-5, 1)]), (4, [(5, 1)])]
    }

    user_matrix, item_features, stream, rewards = linear_batch(**params)
    assert stream.shape == (params['n_samples'], 3)
    print('Reward Matrix')
    print(np.round(rewards, 2))
    print('Best arms\' indices')
    print(rewards.argmax(axis=1))
