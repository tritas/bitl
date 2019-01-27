# -*- coding: utf-8 -*-
# Author: Aris Tritas
# License: BSD 3-clause
from abc import abstractmethod
from abc import ABCMeta


class Policy(metaclass=ABCMeta):
    """ Generic class for policy implementation."""

    def initialize(self):
        """ Initialize the algorithm """
        pass

    @abstractmethod
    def select_arm(self, *args, **kwargs):
        """ Abstract method to be implemented by subclasses """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """ Abstract method to be implemented by subclasses """
        pass
