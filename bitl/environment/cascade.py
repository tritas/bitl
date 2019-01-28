# coding=utf-8
from .base import Environment
from .base import GraphEnvironmentMixin
from .base import ContextualMixin


# TODO(aris): Different types of diffusion processes.


class CascadeEnvironment(GraphEnvironmentMixin, ContextualMixin, Environment):
    """Modeling graph diffusion processes timestep per timestep.
    What interests us is how well we can learn the real influence
    probabilities of one node on another by passive (or active) observation
    of edge activations (i.e. communication)."""

    def __init__(self, *args, **kwargs):
        super(CascadeEnvironment, self).__init__(*args, **kwargs)
