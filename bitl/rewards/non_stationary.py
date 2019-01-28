# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
import traceback

import numpy as np


class VersionedMatrix(object):
    """ Generates a matrix mask for bernouilli rewards
    TODO: Generate rewards from other distributions:
    Dtype object to save normal/expo params instead of Bernouilli mean?
    If non-stationary:
    * have versions of the reward matrix indexed by an auxiliary version id
    associated with each timestep
    * compute the rewards online by using truemodel functions
    """

    def __init__(self, row_blocks, col_blocks, input_matrix):
        self.user_clusters = row_blocks
        self.item_clusters = col_blocks

        self.n_users = sum(row_blocks)
        self.n_items = sum(col_blocks)

        self.breakpoints = None

        dims = (np.sum(row_blocks), np.sum(col_blocks))
        if dims[0] == 0 or dims[1] == 0:
            raise ValueError("Matrix dimensions are malformed")

        rows = np.cumsum(np.hstack(([0], row_blocks)))
        cols = np.cumsum(np.hstack(([0], col_blocks)))
        breakpoints = dict()
        # noinspection PyTypeChecker
        mat_init = np.zeros(dims, dtype=np.float32)

        for i in range(len(row_blocks)):  # ith user group
            for j in range(len(col_blocks)):  # jth item group
                try:
                    # reward distribution mean
                    mu = input_matrix[i][j]["moments"][0]
                    mat_init[
                        rows[i] : rows[i + 1], cols[j] : cols[j + 1]
                    ] = np.full((row_blocks[i], col_blocks[j]), mu)

                    ct = input_matrix[i][j]["breakpoint"]
                    cm = input_matrix[i][j]["future_mean"]
                    if ct and cm:
                        if ct in breakpoints:
                            breakpoints[ct].append(((i, j), cm))
                        else:
                            breakpoints[ct] = [((i, j), cm)]
                except KeyError:
                    traceback.print_exc()
                except BaseException:
                    traceback.print_exc()

        if dims[0] == 1:
            print("Reshaping to remove empty row")
            mat_init = np.reshape(mat_init, dims[1])

        if breakpoints:
            times = [0] + sorted(list(breakpoints.keys()))
            n_versions = len(times)

            if dims[0] != 1:
                full_dims = (n_versions, dims[0], dims[1])
            else:
                full_dims = (n_versions, dims[1])

            time_stamps = np.zeros(times[-1], dtype=np.int32)
            times.append(times[-1])
            reward_mat = np.zeros(full_dims, dtype=np.float32)
            reward_mat[0] = mat_init

            for v in range(1, n_versions):
                reward_mat[v] = reward_mat[v - 1].copy()
                # Set new value for matrix sub-block
                for ((i, j), new_mean) in breakpoints[times[v]]:
                    rslice = slice(rows[i], rows[i + 1])
                    cslice = slice(cols[j], cols[j + 1])
                    if i:
                        reward_mat[v, rslice, cslice] = np.full(
                            (row_blocks[i], col_blocks[j]), new_mean
                        )
                    else:
                        reward_mat[v, cols[j] : cols[j + 1]] = np.full(
                            col_blocks[j], new_mean
                        )
                # Set matrix version number
                # (modifies one more value than is needed,
                # which is reset next round (last index is last revision ofc)
                time_stamps[times[v] - 1 : times[v + 1]] = v
        else:
            print("Stationary rewards")
            reward_mat = mat_init
            time_stamps = np.zeros(1)
        self.reward_mat = reward_mat
        self.time_stamps = time_stamps

    def get_value(self, action, user=None):
        pass

    def set_switch(self, cell, switch_time, new_value):
        pass

    def set_distribution(self, cell, lambda_func):
        pass


class DynamicRewardMatrix(object):
    """
    The main difference with a regular reward matrix is that a non-stationary
    matrix may have rewards that change continuously. This mixin provides
    methods to index rewards by an extra parameter which is time.

    In the most extreme case, each cell is a function of d-dimensional input
    f(x_1, .., x_i, .., x_d; t) to be evaluated upon request
    """

    def __init__(self):
        pass

    def get_reward(self, index_pointer, timestep):
        pass
