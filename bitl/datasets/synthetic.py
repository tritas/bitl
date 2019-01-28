# coding=utf-8
# Author: Aris Tritas
# License: BSD 3-clause

import numpy as np
from sklearn.preprocessing import maxabs_scale

from ..utils.math import noise, sparsify


def contextual_batch(reward_matrix, n_samples):
    """
    Generate a stream of incoming users, their selected items
    and the corresponding reward at each timestep.

    Parameters
    ----------
    reward_matrix : array of shape [n_clusters, n_actions]
        Reward matrix
    n_samples: int

    Returns
    -------
    stream_array : array of shape [n_samples, 3]
       Array of incoming users, items and rewards

    """
    n_users, n_items = reward_matrix.shape
    I = np.random.randint(0, n_users, n_samples)
    J = np.random.randint(0, n_items, n_samples)
    stream_array = np.empty((n_samples, 3), dtype=np.dtype("O"))
    for t in range(n_samples):
        i = I[t]
        j = J[t]
        stream_array[t] = np.array([i, j, reward_matrix[i, j]])
    return stream_array


def linear_batch(n_samples, dims, mixtures, noise_var):
    """ Generate a stream of contextual vectors and the corresponding rewards.
    The reward function is <context, theta> + epsilon,  (bounded in [-1, 1])

    Parameters
    ----------
    n_samples : integer
        Number of samples to generate

    dims : tuple of integers
        The number of users, latent dimension, and number of items to consider

    mixtures : list of tuples
        Parameters for the gaussian distributions generating the user clusters

    noise_var : float
        Level of additive gaussian noise incorporated in the reward function.

    Returns
    -------
    usr_matrix : array of shape [n_users, n_features]
        Each user's contextual vector.

    item_feats : array of shape [n_items, n_features]
        Each item's contextual vector.

    stream_array  : array of shape [n_samples, 4]
        The stream produced by a random process that for each time step,
        chooses a user and an item randomly. Furthermore, the user's context
        and the noise associated is returned to allow comparison of different
        algorithms.

    reward_mat : array of shape [n_users, n_items]
        The matrix which defines for each user u and item v,
        r_uv = <u, v> + noise
    """
    n_users, n_features, n_items = dims

    usr_matrix = gaussian_mixture_matrix(
        (n_users, n_features), mixtures, rescale_clusters=True
    )
    item_features = np.random.randn(n_items, n_features)
    noise_array = np.random.randn(n_users, n_items) * noise_var
    reward_mat = np.dot(usr_matrix, item_features.T) + noise_array
    reward_mat = np.clip(reward_mat, -1, 1)

    I = np.random.randint(0, n_users, n_samples)
    J = np.random.randint(0, n_items, n_samples)
    stream_array = []

    for t in range(n_samples):
        i = I[t]
        j = J[t]
        sample_array = np.array([i, j, noise_array[i][j]])
        stream_array.append(sample_array)

    stream_array = np.asarray(stream_array)

    return usr_matrix, item_features, stream_array, reward_mat


"""
TODO: Two template models
- Heterogenous space of small communities (delicious, reddit etc)
- One large community and some smaller ones (news etc) """


def gaussian_mixture_matrix(dims, tupls_lst, rescale_clusters=False):
    """ Generate a matrix of user contextual vectors belonging to clusters

    Parameters
    ----------

    dims: tuple of integers
        The number of users and the dimension of the latent space

    tupls_lst: list of tuples
        Each tuple has the following form: (ratio, moments) where the ratio
        is the relative size of the cluster, and the moments is a
        list of tuples with the parameters of the normal distrib,
    rescale_clusters: bool

    Returns
    ------
    matrix : array of shape [n_samples, n_features]
        The matrix of contextual vectors


    """
    n_tot_rows, dimension = dims
    matrix = np.empty(dims, dtype=np.float32)
    i2 = 0

    for n_rows, distrib_params_lst in tupls_lst:
        if isinstance(n_rows, float):
            n_rows = int(n_tot_rows * n_rows)

        i1 = i2
        i2 = i1 + n_rows
        # print('Submat users: {} ; start/end indices: <{};{}>'
        # .format(str(n_rows), str(i1), str(i2)))
        submat = np.zeros((n_rows, dimension), dtype=np.float32)
        for (mu, sigma) in distrib_params_lst:
            n = np.random.randn(n_rows, dimension) * sigma + mu
            # print('n_tot_rows({}, {})'.format(str(mu), str(sigma)))
            submat += n
        matrix[i1:i2] = submat
    if rescale_clusters:
        matrix = maxabs_scale(matrix, copy=False)
    return matrix


def stochastic_batch(reward_means, n_samples):
    """Generate a stream for some arm mean reward vector with Bernouilli
    -distributed rewards

     Parameters
     ----------

    reward_means : array of shape [n_actions,]
        The means of each action, distributed as a Bernouilli r.v.
    n_samples: int
    """
    if not reward_means.ndim == 1:
        raise ValueError(
            "Reward vector should be 1-dimensional, got shape {}".format(
                reward_means.shape
            )
        )

    n_items = reward_means.shape[0]
    indices = np.random.randint(0, n_items - 1, n_samples)
    rand_floats = np.random.rand(n_samples)

    stream = np.zeros((n_samples, 2), dtype=np.int32)
    stream[:, 0] = indices
    noisy_draw = reward_means[indices] + np.random.randn(n_samples)
    rewards_mask = noisy_draw < rand_floats
    stream[rewards_mask, 1] = 1
    return stream


def versioned_stochastic_batch(reward_matrix, horizon, versions=()):
    """  Bernouilli arm rewards """
    if not reward_matrix.ndim:
        raise ValueError
    elif reward_matrix.ndim == 1:
        n_items = reward_matrix.shape[0]
        versions = np.zeros(horizon)
    elif reward_matrix.ndim == 2:
        n_versions, n_items = reward_matrix.shape
        # Padding timestamps with last version value
        if len(versions) < horizon:
            padding = np.full(
                horizon - len(versions), versions[-1], dtype=np.int32
            )
            versions = np.hstack((versions, padding))
    else:
        print("Matrix shape is {}".format(reward_matrix.shape))
        # Assuming the user index is on the second dimension
        try:
            n_items, _, n_versions = reward_matrix.shape
            if reward_matrix.shape[1] > 1:
                print(
                    "More than one user for the stochastic setting."
                    "Matrix shape is {}".format(reward_matrix.shape)
                )
                reward_matrix = np.reshape(
                    reward_matrix[:, 0, :], (n_items, n_versions)
                )
        except ValueError as ve:
            raise ve

    items_indx = np.random.randint(0, n_items - 1, horizon)
    uniform_draws = np.random.rand(horizon)

    batch = np.empty((horizon, 3), dtype=np.int32)
    for i in range(horizon):
        # Noisy rewards
        reward = reward_matrix[versions[i], items_indx[i]] + noise()
        if reward < uniform_draws[i]:
            batch[i] = np.array([items_indx[i], versions[i], 0])
        else:
            batch[i] = np.array([items_indx[i], versions[i], 1])
    return batch


def two_item_groups_stream(n_items, horizon, groups_ratio):
    """
    Create a stream with two item groups, simulating news articles.
    The idea is that a large minority of items are relevant to everybody
    (they are asssigned a medium reward), whereas the majority of items
    are specific to some group of people and as such are assigned low reward.
    :param n_items: Total number of items
    :param horizon: Runlength of the stream
    :param groups_ratio: float
        Weight of the `universal` group of items among all items
    :return: stream
       Stream composed, for each timestep, of item id and corresponding reward
    """
    means = two_item_groups(n_items, groups_ratio)
    stream = stochastic_batch(means, horizon)
    return stream


def linear_stream(n_users, n_items, latent_dim, horizon, sparsity):
    """
    Create a stream with contextual vectors for both users and items.
    :param n_users: int
         number of users. If there is a single user, we know
         her contextual vector, it's a waste of memory to pass
         it through the stream at each timestep. TODO: Encapsulate
         stream in array container and pass theta once.
    :param n_items: Total number of items
    :param latent_dim: Contextual space dimension
    :param horizon: Runlength of the stream
    :param sparsity: int, float
        Degree of sparsity (along some dimensions),
        should be an integer and less than the space dimension.
    :return: stream
       Stream composed, for each timestep, of item id and corresponding reward

    """
    U, V, X = latent_linear_bandits(n_users, n_items, latent_dim, sparsity)
    users_seq = np.random.randint(0, n_users - 1, horizon)
    actions_seq = np.random.randint(0, n_items - 1, horizon)
    stream = np.vstack((users_seq, actions_seq)).T
    return stream, U, V


def contextual_matrix_with_base_reward(dims, weights, rewards, noise_std=0.5):
    """ Produce a signal of (vector, action, reward)
    Parameters
    ----------
    dims: tuple
        Shape of the latent context space: (signal size, space dimension)
    weights: numpy array
    rewards: numpy array
    noise_std: float

    Returns
    -------
    signal: array
    """

    horizon, _ = dims
    _, n_actions = rewards.shape

    context = np.random.randn(*dims)
    context_mat = (1 + noise_std) * context
    actions_seq = np.random.randint(0, n_actions - 1, horizon)
    signal = np.empty((horizon, 3), dtype=np.float64)

    for i in range(horizon):
        # Rewards are assumed to have a base value for i
        # given by the reward matrix + the dot product
        # between some contextual vector and a normal gaussian with some noise
        action = actions_seq[i]
        q = np.dot(context_mat[i], weights[action])
        reward = rewards[action] + q
        signal[i] = np.array([context_mat[i], action, reward])
    return signal


def two_item_groups(n_items, groups_ratio=0.4):
    """ Creation of reward weights for:
    a first group of `universal` items with medium mean reward and
    a second group of `specific` items with low mean reward"""

    # Compute index bounds for each group
    n_fst_group = int(round(n_items * groups_ratio))
    n_snd_group = n_items - n_fst_group
    fst_grp_median = int(round(n_fst_group / 2))
    snd_grp_median = n_fst_group + int(round(n_snd_group / 2))
    # Set medium rewards for fst_grp, low reward for snd_grp
    means = np.empty(n_items, dtype=np.float64)
    means[:fst_grp_median] = 0.4
    means[fst_grp_median:n_fst_group] = 0.5
    means[n_fst_group:snd_grp_median] = 0.1
    means[snd_grp_median:] = 0.2
    return means


def latent_linear_bandits(N, M, k, sparsity=0, var=1.0):
    """ Build a random reward matrix from a basis on its rows and columns
    generated from a low-dimensional space.

    Parameters
    ----------
    N: int
        number of rows
    M: int
        number of columns
    k: int
        effective dimension of the space
    sparsity: int, float
        optional sparsity for basis vectors
    var: float
        variance of the normal distribution vectors are drawn from

    Returns
    -------
    U: numpy array of shape [N, k]
    V: numpy array of shape [M, k]
    X: numpy array of shape [N, M]
    """
    # Rows vectors (`context`)
    U = np.random.randn(N, k) * var
    U = sparsify(U, sparsity)
    # Column vectors (`context`)
    V = np.random.randn(M, k) * var
    V = sparsify(V, sparsity)
    # Spectral matrix
    X = U.dot(V.T)

    return U, V, X
