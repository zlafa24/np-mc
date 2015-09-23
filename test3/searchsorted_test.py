#!/usr/bin/python
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

probs = np.array([0.1,0.3,0.2,0.4,0.0])
cumprobs = np.cumsum(probs)
numtrials = 10000
samples = np.empty((numtrials))

for i in xrange(numtrials):
	samples[i] = np.searchsorted(cumprobs,rnd.uniform(0,1))

plt.hist(samples)
plt.show()
