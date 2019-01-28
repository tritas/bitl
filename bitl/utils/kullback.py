# -*- coding: utf-8 -*-
"""Kullback-Leibler utilities"""

from math import exp, log, sqrt

import numpy as np

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.26 $"

# warning: np.dot is miserably slow!

eps = 1e-15


def klBern(x, y):
    """Kullback-Leibler divergence for Bernoulli distributions."""
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))


def klPoisson(x, y):
    """Kullback-Leibler divergence for Poison distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return y - x + x * log(x / y)


def klGamma(x, y, a=1):
    """Kullback-Leibler divergence for gamma distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return a * (x / y - 1 - log(x / y))


def klNegBin(x, y, r=1):
    """Kullback-Leibler divergence for negative binomial distributions."""
    return r * log((r + x) / (r + y)) - x * log(y * (r + x) / (x * (r + y)))


def klGauss(x, y, sig2=0.25):
    """caution modified"""
    """Kullback-Leibler divergence for Gaussian distributions."""
    return (x - y) * (x - y) / (2 * sig2)


def klucb(x, d, div, upperbound, lowerbound=-float("inf"), precision=1e-6):
    """The generic klUCB index computation.

    Input args.:
    x,
    d,
    div:
    KL divergence to be used.
    upperbound,
    lowerbound=-float('inf'),
    precision=1e-6,
    """
    l = max(x, lowerbound)
    u = upperbound
    while u - l > precision:
        m = (l + u) / 2
        if div(x, m) > d:
            u = m
        else:
            l = m
    return (l + u) / 2


def klucbGauss(x, d, sig2=1.0, precision=0.0):
    """klUCB index computation for Gaussian distributions.

    Note that it does not require any search.
    """
    return x + sqrt(2 * sig2 * d)


def klucbPoisson(x, d, precision=1e-6):
    """klUCB index computation for Poisson distributions."""
    # looks safe, to check: left (Gaussian) tail of Poisson dev
    upperbound = x + d + sqrt(d * d + 2 * x * d)
    return klucb(x, d, klPoisson, upperbound, precision)


def klucbBern(x, d, precision=1e-6):
    """klUCB index computation for Bernoulli distributions."""
    upperbound = min(1.0, klucbGauss(x, d))
    # upperbound = min(1.,klucbPoisson(x,d)) # also safe, and better ?
    return klucb(x, d, klBern, upperbound, precision)


def klucbExp(x, d, precision=1e-6):
    """klUCB index computation for exponential distributions."""
    if d < 0.77:
        # safe, klexp(x,y) >= e^2/(2*(1-2e/3)) if x=y(1-e)
        upperbound = x / (1 + 2.0 / 3 * d - sqrt(4.0 / 9 * d * d + 2 * d))
    else:
        upperbound = x * exp(d + 1)
    if d > 1.61:
        lowerbound = x * exp(d)
    else:
        lowerbound = x / (1 + d - sqrt(d * d + 2 * d))
    return klucb(x, d, klGamma, upperbound, lowerbound, precision)


def maxEV(p, V, klMax):
    """Maximize expectation of V wrt. q st. KL(p,q) < klMax.

    Input args.: p, V, klMax.

    Reference: Section 3.2 of [Filippi, Cappé & Garivier - Allerton, 2011].
    """
    Uq = np.zeros(len(p))
    Kb = p > 0.0
    K = ~Kb
    if any(K):
        # Do we need to put some mass on a point where p is zero?
        # If yes, this has to be on one which maximizes V.
        eta = max(V[K])
        J = K & (V == eta)
        if eta > max(V[Kb]):
            y = np.dot(p[Kb], np.log(eta - V[Kb])) + log(
                np.dot(p[Kb], (1.0 / (eta - V[Kb])))
            )
            # print "eta = " + str(eta) + ", y="+str(y);
            if y < klMax:
                rb = exp(y - klMax)
                Uqtemp = p[Kb] / (eta - V[Kb])
                Uq[Kb] = rb * Uqtemp / sum(Uqtemp)
                # or j=min([j for j in range(k) if J[j]]); Uq[j] = r
                Uq[J] = (1.0 - rb) / sum(J)
                return Uq
    # Here, only points where p is strictly positive (in Kb)
    # will receive non-zero mass.
    if any(abs(V[Kb] - V[Kb][0]) > 1e-8):
        eta = reseqp(p[Kb], V[Kb], klMax)  # (eta = nu in the article)
        Uq = p / (eta - V)
        Uq = Uq / sum(Uq)
    # Case where all values in V(Kb) are almost identical.
    else:
        Uq[Kb] = 1 / len(Kb)
    return Uq


def reseqp(p, V, klMax):
    """Solve f(reseqp(p, V, klMax)) = klMax using Newton method.

    Note: This is a subroutine of maxEV.

    Reference: Eq. (4) in Section 3.2 of
    [Filippi, Cappé & Garivier - Allerton, 2011].
    """
    mV = max(V)
    l = mV + 0.1
    tol = 1e-4
    if mV < min(V) + tol:
        return float("inf")
    u = np.dot(p, (1 / (l - V)))
    y = np.dot(p, np.log(l - V)) + log(u) - klMax
    # print "l="+str(l)+", y="+str(y);
    while abs(y) > tol:
        yp = u - np.dot(p, (1 / (l - V) ** 2)) / u  # derivative
        l = l - y / yp
        # print "l = "+str(l)# newton iteration
        if l < mV:
            l = (l + y / yp + mV) / 2  # unlikely, but not impossible
        u = np.dot(p, (1 / (l - V)))
        y = np.dot(p, np.log(l - V)) + log(u) - klMax
        # print "l="+str(l)+", y="+str(y); # function
    return l
