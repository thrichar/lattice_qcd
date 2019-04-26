#!/usr/bin/python

# Code adapted by Thomas H. Richards
# from G. Peter Lepage. "Lattice QCD for Novices."
# In: Proceedings of HUGS 98, edited by J.L. Goity, World Scientific (2000).
# eprint:arXiv:hep-lat/0506036.

import numpy as np
np.random.seed(42)
from math import *
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

def update(x, N, a, eps):
	for j in range(0, N):
		old_x = x[j] # save original value
		old_Sj = S(j, x, N, a)
		x[j] += np.random.uniform(-eps, eps) # update x[j]
		dS = S(j, x, N, a) - old_Sj # change in action

		if dS > 0 and exp(-dS) < np.random.uniform(0, 1):
			x[j] = old_x # restore old value

	return x

def S(j, x, N, a): # harmonic oscillator S
	jp = (j + 1) % N # next site
	jm = (j - 1) % N # previous site

	return a * x[j]**2. / 2. + x[j] * (x[j] - x[jp] - x[jm]) / a

def compute_G(x, n, N):
	g = 0
	for j in range(0, N):
		g += x[j] * x[(j + n) % N]

	return g / N

def MCaverage(x, G, N, N_cor, N_cf, a, eps):
	for j in range(0, N): # initialize x
		x[j] = 0
	for j in range(0, 5*N_cor): # thermalize x
		x = update(x, N, a, eps)
	for alpha in range(0, N_cf):
		for j in range(0, N_cor):
			x = update(x, N, a, eps)
		for n in range(0, N):
			G[alpha][n] = compute_G(x, n, N)

	return G

def deltaE(G, a):
	adE = np.log(np.abs(G[:-1] / G[1:]))

	return adE / a

def main():
	# set parameters:
	N = 100
	N_cor = 20
	N_cf = 1000
	a = 0.1
	eps = 1.4

	# initialize arrays:
	x = np.zeros((N,), np.float32)
	G = np.zeros((N_cf,N), np.float32)

	# do the simulation:
	G = MCaverage(x, G, N, N_cor, N_cf, a, eps)
	G = np.mean(G, axis=0)

	t = np.arange(a, N*a, a)
	deltaEs = deltaE(G, a)
	plt.plot(t, deltaEs, label="Calculated")
	plt.axhline(1., linestyle="--", color="red", label="True")
	plt.xlim(left=0.)
	plt.xlim(right=3.)
	plt.legend(loc="upper left")
	plt.xlabel(r"$t$")
	plt.ylabel(r"$\Delta E(t)$")
	plt.show()
	plt.savefig("deltaE.png")
	plt.clf()

	return

if __name__ == "__main__":
	main()

