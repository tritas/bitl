# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
import importlib.util
from types import FunctionType


def import_from_spec(hook, module_name="bandits", **kwargs):
    """ Returns the class object of a module loaded from a file location """
    filepath, class_name = hook

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class_object = getattr(module, class_name)
    initialized_object = class_object(**kwargs)

    return initialized_object


def policy_name(policy):
    """
    Extract the name of a policy, or use the class name if no
    other information is available. FunctionType is False for Builtin
    functions, use class name.
    :param policy: policy object
    :return: Policy name string
    """
    if isinstance(getattr(policy, '__str__'), FunctionType):
        name = str(policy)
    elif hasattr(policy, '__name__'):
        name = policy.__name__
    else:
        classname_str = str(policy.__class__)
        extract = classname_str.split('.')[-1]
        name = extract[:-2]

    return name
