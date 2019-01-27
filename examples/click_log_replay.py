# -*- coding: utf-8 -*-
# Author: Aris Tritas <aris.tritas@u-psud.fr>
# License: BSD 3-clause
import argparse

from bitl.policies.oful import OFUL

from bitl.datasets.yahoo_click_log import load_clicklog
from bitl.evaluation.bootstraped_evaluation import BootstrapedEvaluation
from bitl.evaluation.loops import yahoo_kernel

parser = argparse.ArgumentParser(description='Click log replay')
parser.add_argument(
    'path', type=str,
    help='enter the path to the yahoo click log compressed file')
parser.add_argument(
    'horizon', type=int, help='stream runlength to use')
args = parser.parse_args()

# Load real-world cleaned and formatted dataset
stream, n_items = load_clicklog(args.path, args.horizon)
features_lst = [features.toarray()[0] for _, _, features, _ in stream]

dimensions = 136
# Define linear (contextual) policies
# lin_ucb = LinUCB(K=n_items, alpha=10, d=dimensions)
oful = OFUL(K=n_items,
            horizon=args.horizon,
            features=features_lst,
            lambda_=1.0,
            delta=0.001,
            d=dimensions,
            C=1.0,
            R=0.5,
            S=0.5)

# Run simulation
evaluator = BootstrapedEvaluation(stream,
                                  yahoo_kernel,
                                  horizon=args.horizon,
                                  n_jobs=-1,
                                  bootstrap=5,
                                  verbose=1)
evaluator.replay(oful)
# evaluator.replay_all([lin_ucb, oful])
