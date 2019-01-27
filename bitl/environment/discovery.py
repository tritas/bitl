# coding=utf-8
# Author: Aris Tritas
# License: BSD 3-clause

from .base import Environment
from .base import GraphEnvironmentMixin


class WikipediaDiscoveryEnvironment(GraphEnvironmentMixin, Environment):
    def step(self, action):
        raise NotImplementedError
