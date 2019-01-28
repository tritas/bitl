# -*- coding: utf-8 -*-
# Author: Aris Tritas
# License: BSD 3-clause
from abc import abstractmethod
from abc import ABCMeta
from types import FunctionType


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


def policy_name(policy):
    """
    Extract the name of a policy, or use the class name if no
    other information is available. FunctionType is False for Builtin
    functions, use class name.
    :param policy: policy object
    :return: Policy name string
    """
    if isinstance(getattr(policy, "__str__"), FunctionType):
        name = str(policy)
    elif hasattr(policy, "__name__"):
        name = policy.__name__
    else:
        classname_str = str(policy.__class__)
        extract = classname_str.split(".")[-1]
        name = extract[:-2]

    return name
