# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
import logging
from time import time

import numpy as np
from networkx import barabasi_albert_graph
from networkx import fast_gnp_random_graph
from networkx import laplacian_matrix
from scipy.sparse.linalg import svds
from scipy.stats import entropy
from sklearn.metrics import pairwise_kernels

logger = logging.getLogger(__name__)


def noise(var=0.1):
    """
    Shorthand for a low level of gaussian noise
    :param var: float
        Variance of the normal distribution to use (default: 0.1)
    :return:
    Scalar realization of the standard normal
    """
    return np.random.randn() * var


def rescale(M):
    """
    Apply feature scaling to a matrix to bring all it's values in [0, 1]
    :param M: The matrix to scale as a numpy.ndarray
    :return: the same ndarray modified in place
    """
    return (M - M.min()) / (M.max() - M.min())


def center(M):
    """
    Normalize an ndarray so that it has zero mean and unit variance
    :param M: The matrix to center as a numpy.ndarray
    :return: the same ndarray modified in place
    """
    return (M - M.mean()) / M.std()


def graph_degree_stats(graph):
    """ Get information about the graph degree distribution as well as
    its moments as a string
    :param graph: A NetworkX Graph object
    :return: stats: str
        A formatted string with graph degree statistics
    """
    degrees = np.array(list(graph.degree().values()))
    stats = "<Min: {}, Max: {}, Mean: {}, Var: {}, Std.Dev:{}>".format(
        degrees.min(),
        degrees.max(),
        degrees.mean(),
        degrees.var().round(2),
        degrees.std().round(2),
    )
    return stats


def coef_to_components(s, n):
    """ Convert percentage to number of components. """
    if isinstance(s, int):
        assert s < n
    elif isinstance(s, float):
        if s < 0 or s > 1:
            raise ValueError(s)
        s = int(s * n)
    else:
        raise TypeError(s)
    return s


def sparsify(X, sparsity):
    """ For each sample, make `sparsity` components null """
    if X.ndim == 1:
        sparsity = coef_to_components(sparsity, len(X))
        sparse_is = np.random.choice(len(X) - 1, size=sparsity, replace=False)
        X[sparse_is] = 0
    elif X.ndim == 2:
        n_samples, n_features = X.shape
        sparsity = coef_to_components(sparsity, n_samples)
        for i in range(n_samples):
            sparse_is = np.random.choice(
                n_features - 1, size=sparsity, replace=False
            )
            X[i, sparse_is] = 0
    else:
        raise NotImplementedError

    return X


def replace_with_noise(matrix, row_sparsity, noise_level):
    """ Replace some means with gaussian noise  """
    assert matrix.ndim == 2, "Only 2D matrix implemented"
    u, v = matrix.shape
    for i in range(u):
        rnoise = np.random.rand(*matrix.shape) * noise_level
        sparse_ind = np.random.choice(v - 1, size=row_sparsity, replace=False)
        matrix[i, sparse_ind] = rnoise[i, sparse_ind]
    return matrix


def kullback_leibler_matrix(X, beta=1, relevance_threshold=0):
    """ Compute the kernel matrix of Kullback-Leibler divergences.
    Remove low-similarity edges by filtering on the exp(-KL) values """
    kl_matrix = pairwise_kernels(X, metric=entropy, n_jobs=-1)
    # Taking the symmetric KL divergence
    kl_matrix += kl_matrix.T
    kl_matrix /= 2
    # XXX: Unsure if we should exponentiate the metric
    kl_matrix_inverse = np.exp(-beta * kl_matrix)
    # Removing self-loops
    kl_matrix_inverse -= np.eye(len(kl_matrix))

    if relevance_threshold:
        mask = np.ma.masked_less(kl_matrix_inverse, relevance_threshold)
        kl_matrix_inverse = np.ma.filled(mask, 0)

    return kl_matrix, kl_matrix_inverse


def erdos_renyi(n, p=None):
    """ Generate a random graph according to the Erdos-Renyi model"""
    p = p or 3 * np.log(n) / n
    return fast_gnp_random_graph(n, p)


def barabasi_albert(n, m):
    """ Generate a Barabasi-Albert graph where n is the number of nodes
    and m is the number of edges used for preferential attachment  """
    return barabasi_albert_graph(n, m)


def graph_laplacian_eig(graph):
    """ Compute a graph laplacian's eigenvalues and eigenvectors.

    Parameters
    ----------
    graph: Graph instance

    Return
    ------
    sigma: eigenvalues
    u: eigenvectors
    """
    logger.info("Computing graph eigendecomposition")
    t0 = time()
    # Compute graph Laplacian
    laplacian = laplacian_matrix(graph)
    # Compute full eigendecomposition
    u, sigma, v = svds(laplacian, k=laplacian.shape[0])
    logger.info("done in {:.3f}s.".format(time() - t0))
    return sigma, u


def fill_reward_matrix(values_tuples, dimensions=1):
    """ Fill matrix with some values (e.g. reward expectation).
    Each value has an arity of n_rows and is copied across the 1st axis.
    """
    rewards = np.hstack(
        [
            np.full((num_rows, dimensions), value, dtype=np.float32)
            for num_rows, value in values_tuples
        ]
    )
    return rewards
