#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.hmm import HMM
import numpy as np

n = 2   # 状态集：H, C
m = 3   # 观测集：1, 2, 3
A = np.array([
    [0.6, 0.4],
    [0.5, 0.5]], dtype=np.float)
B = np.array([
    [0.2, 0.4, 0.4],
    [0.5, 0.4, 0.1]], dtype=np.float)
pi = np.array([0.8, 0.2], dtype=np.float)
obs = np.array([2, 0, 2], dtype=np.int32)
h = HMM(n, m, np.log(A), np.log(B), np.log(pi))

#%% 计算似然度、解码、训练
alpha = h._forward(obs, True)
# print(np.exp(alpha))
# print(np.exp(h.calc_prob(obs)))
beta = h._backward(obs)
# print(np.exp(beta))
v, bt, path = h._viterbi(obs)
# print(np.exp(v))
# print(bt)
# print(path)
# print(h.decode(obs))
xi, gamma = h._baumwelch(obs, alpha, beta)
# print(np.exp(xi))
# print(np.exp(gamma))
model = h._estimate(obs, alpha, beta, xi, gamma)
# print(np.exp(model['A']))
# print(np.exp(model['B']))
# print(np.exp(model['pi']))

#%% 无监督训练
n = 4   # 状态集
m = 5   # 观测集
StateSet = ('VeryCold', 'Cold', 'Hot', 'VeryHot')
ObsSet = (1, 3, 5, 7, 9)
obs = (0, 1, 1, 2, 0, 4, 2, 3, 4, 2, 0, 2, 3, 4, 4, 2, 1, 0, 1) * 10

h = HMM(n, m)
h.reset('random')
# h.reset('uniform')
h.train(np.array(obs), 100)
# h.save('h.hdf5')
print(np.exp(h.pi))
print(np.exp(h.A))
print(np.exp(h.B))

# obs = (1, 0, 2, 3, 2, 0, 0, 1, 0, 1)
# h = HMM()
# h.load('h.hdf5')
# for k in h.decode(obs):
#     print(StateSet[k], end=' ')
# print(np.array(h.pi))
# print(np.array(h.A))
# print(np.array(h.B))
