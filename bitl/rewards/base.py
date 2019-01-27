import numpy as np


class RewardMatrix(object):
    """
    The reward matrix holds the reward distributions of all states
    for a certain problem.
    """

    def __init__(self, dimensions):
        """
        Parameters
        ----------

        dimensions : tuple of int
            The shape of the reward distribution
        """
        self._rewards = np.zeros(dimensions, dtype=np.float64)

    def get_reward(self, index_pointer):
        """
        :param index_pointer:
        """
        return self._rewards[index_pointer]