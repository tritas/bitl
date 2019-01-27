# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
from time import time

import numpy as np
from numpy import clip, empty
from numpy.random import binomial, normal, randint

from ..utils.math import noise

"""
An idea would be to give each kernel hooks to the evaluator/environment
so that it can register its results and query for reward information
(as well as other side-information)
"""

"""
def linear_contextual_kernel(policy, stream, horizon):
    for t in range(horizon):
        features, usr, arm, nz = stream[ind[t]]
        usr = int(usr)
        arm = int(arm)

        action_t = policy.select_arm(features, usr)
        reward_t = rm[l[t]][usr, arm]
        policy.update(action_t, reward_t + nz, features, usr)

        reward[sim, t] = reward_t
        regret[sim, t] = rm[l[t]][usr].max() - reward_t


def linear_kernel(policy, stream, horizon):
    for t in range(horizon):
        features, usr, arm, nz = stream[ind[t]]
        usr = int(usr)
        arm = int(arm)

        action_t = policy.select_arm(features, usr)
        policy.update(action_t, rm[l[t]][action_t] + nz, features, usr)

        reward[sim, t] = rm[l[t]][action_t] + nz
        regret[sim, t] = rm[l[t]].max() - rm[l[t]][action_t]


def contextual_kernel(policy, stream, horizon):
    for t in range(horizon):
        usr, arm, _reward = stream[ind[t]]
        usr = int(usr)
        arm = int(arm)

        action_t = policy.select_arm(usr)
        reward_t = binomial(1, rm[l[t]][usr, action_t] + noise())
        policy.update(usr, action_t, reward_t)

        reward[sim, t] = reward_t
        regret[sim, t] = rm[l[t]][usr].max() - policy.means[usr, action_t]
"""


def adversarial_kernel(policy, stream, reward_matrix, horizon):
    """

    :param policy:
    :param stream:
    :param reward_matrix:
    :param horizon:
    :return:
    """
    optimal_mean = reward_matrix.max()
    reward = empty(horizon)
    regret = empty(horizon)

    for t in range(horizon):
        adversary_arm, v, reward_t = stream[t]

        action_t = policy.select_arm()
        policy.update(action_t, reward)

        reward[t] = reward_t
        regret[t] = optimal_mean - reward_matrix[action_t]

    return reward, regret


def base_kernel(policy, signal, reward_matrix, horizon):
    """ Bernouilli rewards and a single user """
    optimal_mean = reward_matrix.max()
    reward = empty(horizon)
    regret = empty(horizon)

    for t in range(horizon):
        action_t = policy.select_arm()
        reward_t = binomial(1, reward_matrix[action_t]) + noise()
        reward_t = clip(reward_t, 0, 1)
        policy.update(action_t, reward_t)

        reward[t] = reward_t
        regret[t] = optimal_mean - reward_matrix[action_t]

    return reward, regret


def yahoo_kernel(policy, signal, horizon, jitter, **side_info):
    n_valid = 0
    payoff = 0
    policy.initialize(**side_info)
    for i in range(horizon * policy.n_arms):
        ind = randint(horizon)
        item, reward, features, _ = signal[ind]
        feat_vect = features.toarray()[0].astype(float)
        # feat_vect, usr, item, reward = signal[ind]

        if jitter:
            feat_vect += normal(0, jitter, feat_vect.shape)

        predicted_item = policy.select_arm(feat_vect)

        if item == predicted_item:
            n_valid += 1
            payoff += reward

        policy.update(item, reward)
    return n_valid, payoff


"""
def versioned_base_kernel(policy, stream, reward_matrix, horizon):
    ind = arange(horizon)
    random.shuffle(ind)
    for t in range(horizon):
        _arm, v, reward = stream[ind[t]]
        action_t = policy.select_arm()

        reward_t = binomial(1, rm[v][action_t] + noise())
        policy.update(action_t, reward_t)

        optimal_mean = rm[v].max()
        rewards[sim, t] = reward_t
        regret[sim, t] = optimal_mean - rm[v][action_t]
"""


def bootstrap_with_side_information(policy, signal, env, horizon, bootstrap):
    """

    :param policy:
    :param signal:
    :param env:
    :param horizon:
    :param bootstrap:
    """
    n_valid = np.zeros(bootstrap, dtype=np.int32)
    payoff = np.zeros(bootstrap)
    for b in range(bootstrap):
        policy.initialize()
        for i in range(horizon * env.n_actions):
            ind = randint(horizon)
            # Need to unpack one-by-one as python2 does not support
            # start unpack (starred expression assignment)
            item = signal[ind][0]
            reward = signal[ind][1]
            side_information = signal[ind][2:]
            policy_item = policy.choose(*side_information)
            if item == policy_item:
                n_valid[b] += 1
                payoff[b] += reward
            policy.update(item, reward)


def discovery_kernel(agent, n_runs, horizon, is_interesting, step_features,
                     theta_star):
    """

    :param agent:
    :param n_runs:
    :param horizon:
    :param is_interesting:
    :param step_features:
    :param theta_star:
    :return:
    """
    n_found = np.zeros((n_runs, horizon), dtype=np.uint8)
    runtime = np.zeros(n_runs, dtype=np.float64)
    for run in range(n_runs):
        t0 = time()
        agent.initialize()

        for t in range(horizon):
            item = agent.choose()

            if item is not None and is_interesting[item]:
                n_found[run, t] = 1
            elif item is None:
                continue

            rm = np.dot(theta_star, step_features[t]) + noise()
            # Rewards must be in [-1, 1] for the GaussianTS prior update
            rm = np.clip(rm, -1, 1)
            agent.update(item, rm)
        runtime[run] = time() - t0

    found = n_found.mean(axis=0).cumsum()
    return found, runtime.mean()
