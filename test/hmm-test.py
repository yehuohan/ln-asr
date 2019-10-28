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
h = HMM(n, m, A, B, pi)

#%% 计算似然度、解码、训练
alpha = h._forward(obs, True)
# print(alpha)
# print(h.calc_prob(obs))
beta = h._backward(obs)
# print(beta)
v, bt, path = h._viterbi(obs)
# print(v)
# print(bt)
# print(path)
# print(h.decode(obs))
xi, gamma = h._baumwelch(obs, alpha, beta)
model = h._estimate(obs, alpha, beta, xi, gamma)
# print(model['A'])
# print(model['B'])
# print(model['pi'])

#%% 无监督训练
obs = (3,1,2,1,0,1,2,3,1,2,0,0,0,1,1,2,1,3,0) * 10

atmp = np.random.random_sample((4, 4))
a = atmp / atmp.sum(axis=1)[:, np.newaxis]    
btmp = np.random.random_sample((4, 4))
b = btmp / btmp.sum(axis=1)[:, np.newaxis]
pitmp = np.random.random_sample((4))
pi = pitmp / np.sum(pitmp)

h = HMM(4, 4, a, b, pi)
h.train(np.array(obs), 100)
print(h.pi)
print(h.A)
print(h.B)
