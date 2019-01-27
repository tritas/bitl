# -*- coding: utf-8 -*-
# Author: Aris Tritas, Nithin Holla
# License: BSD 3-clause

import numpy as np
from scipy.optimize import minimize


class OFUL(object):
    # Constructor function
    """
    Optimism-in-the-face-of-Uncertainty linear bandits
    """

    def __init__(self, n_actions, action_features, horizon, dimensions, lambda_=1.0, delta=0.001, C=1.0, R=0.5, S=0.5):
        self.action_features = np.asarray(action_features)
        self.n_actions = n_actions
        self.n_features = dimensions

        self.horizon = horizon
        self.lambda_ = lambda_
        self.delta = delta
        self.C = C
        self.R = R
        self.s = S
        self.S = None
        self.t = None
        self.n_draws = None
        self._means = None
        self._theta_star = None
        self._Vt = None  # Matrix Vt
        self._detVt = None  # Determinant of Vt
        self._detV_tau = None

        self._features = None
        self._rewards = None

    # Initialize the class variables
    def initialize(self):
        """ Reset """
        self._detV_tau = 1
        self.S = self.s
        self.t = 0

        self.n_draws = np.zeros(self.n_actions, dtype=np.int32)
        self._means = np.zeros(self.n_actions, dtype=np.float64)
        self._Vt = self.lambda_ * np.identity(self.n_features)
        self._detVt = np.linalg.det(self._Vt)

        self._theta_star = np.zeros(self.n_features, dtype=np.float64)
        self._features = np.zeros((0, self.n_features), dtype=np.float64)
        self._rewards = np.zeros((0, 1), dtype=np.float64)

    def select_arm(self, *args):
        """ Choose the best arm as per OFUL """
        for i in range(self.n_actions):
            if self.n_draws[i] == 0:
                return i

        # Compute theta_hat
        temp1 = np.dot(self._features.T, self._features)
        temp2 = np.linalg.inv(temp1 + self.lambda_ * np.eye(len(temp1)))
        temp3 = np.dot(temp2, self._features.T)
        theta_hat = np.dot(temp3, self._rewards)
        theta_hat = np.ravel(theta_hat)

        # Compute Vt
        self._Vt += temp1
        self._detVt = np.linalg.det(self._Vt)

        # Saving computation
        sparsity_cond = np.sum(self._theta_star == 0) == self.n_features
        det_growth_cond = self._detVt > (1 + self.C) * self._detV_tau
        if sparsity_cond or det_growth_cond:
            # Compute the ellipsoidal bound centered at theta_hat
            temp = np.linalg.det(self.lambda_ * np.eye(self.t))
            log_det = np.sqrt(2 * np.log((np.sqrt(self._detVt) * temp) / self.delta))
            bound = self.R * log_det + np.sqrt(self.lambda_) * self.S

            # Optimize the objective function and return the best arm
            max_obj = -np.infty
            obj = np.zeros(self.n_actions)
            for arm in range(self.n_actions):
                x = self.action_features[arm]
                opt_args = (self._Vt, (theta_hat - self._theta_star), bound)
                cons = dict(type="ineq", fun=cons_func, args=opt_args)

                opt_res = minimize(
                    fun=obj_func,
                    x0=theta_hat,
                    args=(x, self._theta_star),
                    method="SLSQP",
                    constraints=cons,
                )
                theta_est = opt_res["x"]
                obj[arm] = np.dot(theta_est, x)

                if obj[arm] > max_obj and obj[arm] != 0:
                    self._theta_star = theta_est
                    max_obj = obj[arm]
                    self._detV_tau = self._detVt
                    self.S = np.sqrt(np.dot(self._theta_star.T, self._theta_star))

            return np.argmax(obj)
        else:
            return np.array(
                [
                    np.dot(self._theta_star, self.action_features[arm])
                    for arm in range(self.n_actions)
                ]
            ).argmax()

    def update(self, arm, reward):
        """ Update the class variables after observation of reward """
        self.t += 1
        self.n_draws[arm] += 1
        n = self.n_draws[arm]
        self._means[arm] = ((n - 1) / float(n)) * self._means[arm] + (
            1 / float(n)
        ) * reward

        self._features = np.vstack((self._features, self.action_features[arm]))
        self._rewards = np.vstack((self._rewards, reward))


def cons_func(x, A, v, b):
    """ Compute the constraint function for inequality
    Difference between the bound and the weighted norm """
    return b - np.sqrt(np.dot(np.dot(v.T, A), v))


def obj_func(x, x1, x2):
    """ Compute objective function - scalar product """
    # Objective function is -value to solve maximization problem
    # using minimize function
    return np.dot(x1.T, x2) * -1
