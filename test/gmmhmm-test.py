#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.gmmhmm import GMMHMM
import numpy as np

n = 5   # 状态集
m = 3   # 观测集
d = 2   # 观测数据维度
T = 40

obs = np.array((0.6 * np.random.random_sample((T, d)) - 0.3), dtype=np.double)

gh = GMMHMM(n, m, d)
# gh.reset('uniform')
gh.reset('random')
gh.train(obs, 10)
gh.save('gh.hdf5')
print("Pi\n", np.exp(gh.pi))
print("A\n", np.exp(gh.A))
print("weights\n", np.exp(gh.w))
print("mu\n", gh.mu)
print("sigam\n", gh.si)

# gh = GMMHMM()
# gh.load('gh.hdf5')
# print(gh.decode(obs))
