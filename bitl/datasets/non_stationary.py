# -*- coding: utf-8 -*-
# Non-stationary Sequential Prediction with Confidence
# Copyright (c) 2016, Odalric-Ambrym Maillard
# Organization: Inria Saclay Ile de France

from numpy import dot, log, sin, sqrt
from numpy.random import rand, randint, randn


# Definition of a few models
def truemodel1(x):
    return 1, x


def truemodel2(x):
    return 1, x, x * x


def truemodel3(x):
    return 1, log(1 + x), sqrt(x), x


def truemodel4(x):
    return 1, x, x * x, sin(x / 10.)


def truemodel5(x):
    return 1, x, sin(x / 10.), sin(x / 20.)


def truemodel6(x):
    return 1, sin(6 * x / 5.), sin(6 * x / 11.)


def truemodel7(x):
    return 1, sin(6 * x / 5.), sin(6 * x / 11.), sin(6 * x / 19.)


# ----------------------------------------------------------------------------
#  Signal Generation
# ----------------------------------------------------------------------------
def non_stationary_signal(NumberOfPieces=5,
                          MinLengthofPiece=30,
                          MaxLengthofPiece=50):
    # (20, 20, 40)
    # Parameter: number of pieces, minimum  and maximum length  of a piece
    piecewiseModels = []
    startChange = 0
    changeTimes = []
    # Construction of the pieces
    for change in range(0, NumberOfPieces):
        if change > 1:  # >2
            # (6,20) # Selects randomly one model, or a previously chosen one.
            mode = randint(8, 20)
        else:
            mode = randint(6, 8)  # (1,8)
            result = [startChange, truemodel1,
                      (randn(), randn() / 2),
                      randint(1, 3) * rand()]
        if mode == 1:
            result = [startChange, truemodel1,
                      (randn(), randn() / 2),
                      randint(1, 3) * rand()]
        if mode == 2:
            result = [startChange, truemodel2,
                      (randn(), randn() / 2, randn() / 6),
                      randint(1, 3) * rand()]
        if mode == 3:
            result = [startChange, truemodel3,
                      (randn(), randn() / 2, randn() / 6, randn() / 24),
                      randint(1, 3) * rand()]
        if mode == 4:
            result = [startChange, truemodel4,
                      (randn(), randn() / 2, randn() / 24, 10 * randn()),
                      randint(1, 3) * rand()]
        if mode == 5:
            result = [startChange, truemodel5,
                      (randn(), randn() / 2, 10 * randn(), 10 * randn()),
                      randint(1, 3) * rand()]
        if mode == 6:
            result = [startChange, truemodel6,
                      (randn(), randn() / 2, 10 * randn()),
                      randint(1, 3) * rand()]
        if mode == 7:
            result = [startChange, truemodel7,
                      (randn(), randn() / 2, 10 * randn(), 10 * randn()),
                      randint(1, 3) * rand()]
        if (mode >= 8) and (change > 1):
            cpast = randint(0, change)
            result = [startChange, piecewiseModels[cpast][1],
                      piecewiseModels[cpast][2], piecewiseModels[cpast][3]]

        piecewiseModels.append(result)
        startChange += randint(MinLengthofPiece, MaxLengthofPiece)
        changeTimes.append(startChange)

    MaxNumberOfObservations = startChange - 1  # MaxLengthofPiece
    # Construction of the signal
    # Parameter: number of desired observations
    # ----------------------------------------------------------------------------
    # What we have generated is:
    # XObservationPoints contains the x points,
    # YObservationValues  contains the y values,
    # FObservationValues contains the  f values, such as y(x) = f(x)+\xi.
    # ----------------------------------------------------------------------------
    NumberOfObservations = min(150, MaxNumberOfObservations)
    XObservationPoints = []
    FObservationValues = []
    YObservationValues = []
    change = 0
    for x in range(0, NumberOfObservations):
        XObservationPoints.append(x)
        if change + 1 < NumberOfPieces:
            if x >= piecewiseModels[change + 1][0]:
                change += 1
        [startChange, trueModel, trueModelParameter, sigma] = \
            piecewiseModels[change]
        f = dot(trueModelParameter, trueModel(x - startChange))

        FObservationValues.append(f)
        xi = randn() * sigma
        YObservationValues.append(f + xi)

    return XObservationPoints, YObservationValues, FObservationValues
