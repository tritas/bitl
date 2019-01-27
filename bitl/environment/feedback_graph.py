# -*- coding: utf-8 -*-
# Author: Aris Tritas
# License: BSD 3-clause


class FeedbackGraph:
    """
    A feedback graph shares the result of a loss function between nodes
    (experts) according to dependence relations represented by the edges
    between nodes. At each timestep, results are propagated to the observation
    system defined by each expert.
    """

    def __init__(self, experts, graph):
        """ Make a feedback graph object.
        Parameters
        ---------
        experts : dictionary of (policy name, policy object)
            The experts that will be nodes on the graph

        graph : graph object
            Directed or undirected graph encoding dependencies between experts
        """
        self.experts_dict = experts
        self.feedback_graph = graph
        assert len(self.experts_dict) == len(self.feedback_graph.nodes()), \
            'Incompatible number of nodes and experts'

    def propagate(self, source, loss):
        """Update each related expert with the loss given to the source

        Parameters
        ----------

        source : policy object
            The expert observing the loss

        loss : float
            Value of the loss function
        """
        observation_system = self.feedback_graph.edges[source]
        if not len(observation_system):
            return
        for edge in observation_system:
            self.experts_dict[edge].update(loss)
