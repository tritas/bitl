# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
from abc import ABCMeta
from abc import abstractmethod

import numpy as np


class Environment(metaclass=ABCMeta):
    """
    An object to define the structured of users and items.
    For example we may want to define some clustering,
    or some contextual information associated with items.
    """

    def __init__(self, n_nodes):
        self.clustering = dict()
        self.latent_dimension = 0

    def add_cluster(self, n_nodes):
        """ Add a cluster on users or items? """
        n_clusters = len(self.clustering) + 1
        self.clustering[n_clusters] = n_nodes

    @abstractmethod
    def step(self, action):
        """Use known (state, action) to generate a tuple of
        (reward, new state).
        This could also be: 'obs', 'rew', 'game_over', 'info'.
        """
        pass


class GraphEnvironmentMixin(object):
    """Gives access to methods that can grow the graph dynamically,
    as well as structural properties of the object such as the Laplacian
    and its eigendecomposition."""

    def __init__(self, graph):
        """Initialize the graph structure properties.

        Parameters
        ----------
        graph: object
          Anything that can be imported by the graph library we use
        """
        self.graph = graph
        self.n_nodes = None
        self.n_edges = None


class TreeEnvironmentMixin(GraphEnvironmentMixin):
    """If you ever want to implement MCTS"""

    pass


class ClusterMixin(object):
    """Both the users and action can admit some underlying cluster/dendogram
    structure."""

    def __init__(self, cluster_idx):
        """Initialize the structure with an array of cluster membership."""
        self.cluster_idx = cluster_idx
        self.num_clusters = np.unique(cluster_idx)


class HierarchicalClusterMixin(ClusterMixin):
    """ Model a hierarchical latent structure. """

    def __init__(self, dendogram):
        """Initialize the structure with an array of hierarchy membership."""
        self.dendogram = dendogram
        # TODO: Specify which part of the dendogram has the cluster indices.
        super(HierarchicalClusterMixin, self).__init__()


class ContextualMixin(object):
    """
    User/Item environment with some structure.
    add contextual information (i.e. ontological, geo-spatial etc.)
    """

    def __init__(self, n_items=0):
        pass


class DelayedRewardMixin(object):
    pass


class NonStationaryRewardMixin(object):
    pass


class RankedItemsMixin(object):
    # TODO: Imagine we have to simulate ranked item sets
    pass
