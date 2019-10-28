#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.gmmhmm import GMMHMM
import numpy as np

n = 5
m = 4
d = 2
T = 40

atmp = np.random.random_sample((n, n))
a = np.array(atmp / atmp.sum(axis=1)[:, np.newaxis], dtype=np.double)    

wtmp = np.random.random_sample((n, m))
w = np.array(wtmp / wtmp.sum(axis=1)[:, np.newaxis], dtype=np.double)

mu = np.array((0.6 * np.random.random_sample((n, m, d)) - 0.3), dtype=np.double)

Sigma = np.zeros((n, m, d, d), dtype=np.double)
Sigma[:, :] = np.eye((d), dtype=np.double)

pitmp = np.random.random_sample((n))
pi = np.array(pitmp / np.sum(pitmp), dtype=np.double)

obs = np.array((0.6 * np.random.random_sample((T, d)) - 0.3), dtype=np.double)

gh = GMMHMM(n, m, d, a, pi, w, mu, Sigma)
gh.train(obs, 10)
print("Pi\n", gh.pi)
print("A\n", gh.A)
print("weights\n", gh.w)
print("mu\n", gh.mu)
print("sigam\n", gh.si)
