# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause


class Metrics:
    def __init__(self):
        pass

    def aggregate(self, metric, op):
        pass

    def output(self):
        pass


class BREDMetrics(Metrics):
    def __init__(self):
        super().__init__()

    def aggregate(self, metric, op): pass

    def output(self): pass


class BaseMetrics(Metrics):
    def __init__(self):
        super().__init__()

    def aggregate(self, metric, op): pass

    def output(self): pass
