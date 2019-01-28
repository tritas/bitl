# coding=utf-8
import numpy as np
from bitl.utils.kullback import klucbBern
from bitl.utils.kullback import klucbExp
from bitl.utils.kullback import klucbGauss
from bitl.utils.kullback import klucbPoisson
from bitl.utils.kullback import maxEV
from bitl.utils.kullback import reseqp

# from matplotlib.pyplot import *
# t = linspace(0, 1)
# subplot(2, 1, 1)
# plot(t, kl(t, 0.6))
# subplot(2, 1, 2)
# d = linspace(0, 1, 100)
# plot(d, [klucb(0.3, dd) for dd in d])
# show()
print(klucbGauss(0.9, 0.2))
print(klucbBern(0.9, 0.2))
print(klucbPoisson(0.9, 0.2))
p = np.array([0.3, 0.5, 0.2])
p = np.array([0., 1.])
V = np.array([10, 3])
klMax = 0.1

p = np.array(
    [0.11794872, 0.27948718, 0.31538462, 0.14102564,
     0.0974359, 0.03076923, 0.00769231, 0.01025641, 0.])
V = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])
klMax = 0.0168913409484

print("eta = " + str(reseqp(p, V, klMax)))
print("Uq = " + str(maxEV(p, V, klMax)))

x = 2
d = 2.51
print("klucb = " + str(klucbExp(x, d)))
ub = x / (1 + 2. / 3 * d - np.sqrt(4. / 9 * d * d + 2 * d))
print("majoration = " + str(ub))
print("maj bete = " + str(x * np.exp(d + 1)))
