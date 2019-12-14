#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
隐马尔可夫模型（Hidden Markov Model, HMM）
"""

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.utils import lse
from typing import Tuple
import numpy as np
import h5py

class HMM:
    """
    HMM基本模型，形式化定义如下：

    - n: 隐状态数，状态集为Q={q1, q2, ... ,qN}
    - m: 观测集数量，观测集为V={v1, v2, ... ,vM}，一个实际的观测序列为O={o1, o2, ... oT}
    - A: 状态转移矩阵(NxN)，A[i, j]表示状态i转移到状态j
    - B: 发射概率(NxM)，B[i, ot]表示状态i生成观测ot概率（观测ot时状态为i的概率）
    - pi: 初始概率分布(Nx1)
    - obs: 一个观测度序列O(T)

    A, B, pi等变量的概率值使用log概率，对数底为e。
    故对于forward, backward等算法，公式形式有所变化：
    - np.zeros: 改成-np.inf，因为log(0)=-inf
    - np.ones: 改成np.exp(np.ones)
    - np.multiply: 改成加法，log(a*b)=log(a)+log(b)
    - np.divide: 必成减法，log(a/b)=log(a)-log(b)
    - np.sum: 改成log-sum-exp，log(log(a) + log(b))=log(e**log(a) + e**log(b))
    """

    def __init__(self, n:int=1, m:int=1,
            A:np.matrix=None, B:np.matrix=None, pi:np.ndarray=None,
            precision=np.double):
        """初始化HMM参数"""
        self.n = n
        self.m = m
        self.A = A
        self.B = B
        self.pi = pi
        self.precision = precision

    def _map_b(self, obs:np.ndarray):
        """映射发射概率

        B与b的映射关系：

        - B: (NxM), B[i, obs[t]]
        - b: (NxT), b[i, t]

        :Parameters:
            - obs: 观测序列O={o1, o2, ... oT}，值为观测集中的值
        """
        T = len(obs)
        self.b = np.empty((self.n, T), dtype=self.precision)
        for t in np.arange(T):
            self.b[:, t] = self.B[:, obs[t]]

    def _forward(self, obs:np.ndarray, remap:bool=False)->np.ndarray:
        """向前算法

        ::

            在t,j处alpha网格计算示意图：
                q1  q2 ... qN
            o0
            ... #   #       #   -> alpha[t-1], A[:, j]
            ot  t,j             -> B[j, ot]
            ...
            oT

        :Parameters:
            - obs: 观测序列O={o1, o2, ... oT}，值为观测集中的值
            - remap: 是否重新映射发射概率

        :Returns: 向前传递因子alpha(TxN)，也称向前网格、向前概率
        """
        if remap:
            self._map_b(obs)
        # 低空间复杂度版本（只能保存最后一行alpha的值）
        # alpha = self.pi + self.b[:, 0]
        # p = np.copy(alpha)
        # for t in range(1, len(obs)):
        #     for j in range(self.n):
        #         alpha[j] = lse(p + self.A[:, j]) + self.b[j, t]
        #     p = np.copy(alpha)
        # return alpha
        T = len(obs)
        alpha = np.empty((T, self.n), dtype=self.precision)
        # 初始状态
        alpha[0] = self.pi + self.b[:, 0]
        # 迭代
        for t in np.arange(1, T):
            for j in np.arange(self.n):
                alpha[t, j] = lse(alpha[t-1] + self.A[:, j]) + self.b[j, t]
        return alpha

    def _backward(self, obs:np.ndarray, remap:bool=False)->np.ndarray:
        """向后算法（类似于forward）

        ::

            在t,i处beta网格计算示意图：
                q1  q2 ... qN
            o0
            ...
            ot  t,i
            ... #   #       #   -> beta[t+1], A[i:, :], B[j, o(t+1)]
            oT

        :Parameters:
            - obs: 观测序列O={o1, o2, ... oT}，值为观测集中的值
            - remap: 是否重新映射发射概率

        :Returns: 向后传递因子beta(TxN)，也称向后网格、向后概率
        """
        if remap:
            self._map_b(obs)
        T = len(obs)
        beta = np.empty((T, self.n), dtype=self.precision)
        # 初始状态
        beta[-1] = np.log(np.ones(self.n, dtype=self.precision))
        # 迭代
        for t in np.arange(T-2, -1, -1):
            for i in np.arange(self.n):
                beta[t, i] = lse(beta[t+1] + self.A[i, :] + self.b[:, t+1])
        return beta

    def _viterbi(self, obs:np.ndarray, remap:bool=False)->Tuple[np.ndarray]:
        """Viterbi算法

        ::

            在t,j处viterbi网格计算示意图：
                q1  q2 ... qN
            o0
            ... #   #       #   -> v[t-1], A[:, j]
            ot  t,j             -> B[j, ot], bt[t, j]
            ...
            oT

            计算出viterbi网格后，才能知道oT对应的最大概率的状态，进而通过backtrace找到最大概率序列

        :Parameters:
            - obs: 观测序列O={o1, o2, ... oT}，值为观测集中的值
            - remap: 是否重新映射发射概率

        :Returns: Viterbi传递因子（Viterbi网格，Viterbi概率），反追踪网格，最大概率的状态序列路径
        """
        if remap:
            self._map_b(obs)
        T = len(obs)
        v = np.empty((T, self.n), dtype=self.precision)
        bt = np.zeros((T, self.n), dtype=np.uint32)
        # 初始状态
        v[0] = self.pi + self.b[:, 0]
        # 迭代
        for t in np.arange(1, T):
            for j in np.arange(self.n):
                val = v[t-1] + self.A[:, j]
                v[t, j] = np.max(val) + self.b[j, t]
                bt[t, j] = np.argmax(val)
        # 反向追踪(backtrace)
        path = np.empty(T, dtype=np.uint32)
        path[-1] = np.argmax(v[-1])
        for t in np.arange(T-2, -1, -1):
            path[t] = bt[t+1, path[t+1]]
        return (v, bt, path)

    def _baumwelch(self, obs:np.ndarray, alpha:np.ndarray, beta:np.ndarray, remap:bool=False)->Tuple[np.ndarray]:
        """Baum-Welch算法（向前向后算法）

        :Parameters:
            - obs: 观测序列O={o1, o2, ... oT}，值为观测集中的值
            - alpha: 向前传递因子(TxN)
            - beta: 向后传递因子(TxN)
            - remap: 是否重新映射发射概率

        :Returns: xi和gamma参数
        """
        if remap:
            self._map_b(obs)
        T = len(obs)
        # E-Step: xi(TxNxN), gamma(TxN)
        # xi[t][i][j]
        xi = np.full((T, self.n, self.n), -np.inf, dtype=self.precision)
        for t in np.arange(T-1):
            numer = alpha[t, :].reshape((self.n, 1)) \
                    + self.A \
                    + self.b[:, t+1].reshape((1, self.n)) \
                    + beta[t+1, :].reshape((1, self.n))
            denom = lse(numer)
            xi[t] = numer - denom
        # gamma[t][i]: 对xi的j求和
        gamma = lse(xi, axis=2)
        return xi, gamma

    def _estimate(self, obs:np.ndarray, alpha, beta, xi, gamma)->dict:
        """计算Maximization模型新参数

        :Parameters:
            - obs: 观测序列O={o1, o2, ... oT}，值为观测集中的值
            - alpha: 向前传递因子(TxN)
            - xi: baum-welch参数(TxNxN)
            - gamma: baum-welch参数(TxN)

        :Returns: HMM模型参数(A,B,pi)
        """
        # M-Step: new A,B,pi
        denom = lse(gamma, axis=0)
        A = lse(xi, axis=0) - denom.reshape((self.n, 1))
        B = np.empty((self.n, self.m), dtype=self.precision)
        for k in np.arange(self.m):
            B[:, k] = lse(gamma[k==obs, :], axis=0) - denom.reshape((1, self.n))
        pi = gamma[0]
        # New model
        model = {}
        model['A'] = A
        model['B'] = B
        model['pi'] = pi
        return model

    def _updatemodel(self, model:dict):
        """更新模型参数"""
        self.A = model['A']
        self.B = model['B']
        self.pi = model['pi']

    def reset(self, init_type):
        """初始化训练参数

        :Parameters:
            - init_type: 初始化类型，默认全部初始化为零
        """
        if init_type == 'uniform':
            self.A = np.log(np.ones((self.n, self.n), dtype=self.precision) * (1.0 / self.n))
            self.B = np.log(np.ones((self.n, self.m), dtype=self.precision) * (1.0 / self.m))
            self.pi = np.log(np.ones(self.n, dtype=self.precision) * (1.0 / self.n))
        elif init_type == 'random':
            # random范转(0, 1.0]，防止出现log(0)
            a = 1.0 - np.random.random_sample((self.n, self.n))
            b = 1.0 - np.random.random_sample((self.n, self.m))
            pi = 1.0 - np.random.random_sample((self.n))
            self.A = np.log(a / a.sum(axis=1)[:, np.newaxis])
            self.B = np.log(b / b.sum(axis=1)[:, np.newaxis])
            self.pi = np.log(pi / np.sum(pi))

    def save(self, filename):
        """保存模型参数"""
        f = h5py.File(filename, 'w')
        f.create_dataset('A', data=self.A)
        f.create_dataset('B', data=self.B)
        f.create_dataset('pi', data=self.pi)
        f.close()

    def load(self, filename):
        """加载模型参数"""
        f = h5py.File(filename, 'r')
        self.A = f['A']
        self.B = f['B']
        self.pi = f['pi']
        self.n, self.m = self.B.shape

    def calc_prob(self, obs:np.ndarray)->float:
        """计算观测序列的log似然度

        :Parameters:
            - obs: 观测序列O={o1, o2, ... oT}，值为观测集中的值
        """
        return lse(self._forward(obs, True)[-1])

    def decode(self, obs:np.ndarray)->np.ndarray:
        """解码最可能的隐藏状态序列

        :Parameters:
            - obs: 观测序列O={o1, o2, ... oT}，值为观测集中的值
        """
        _, _, path = self._viterbi(obs, True)
        return path

    def train(self, obs, iters=1, eps=0.0001, verbose:bool=True):
        """训练HMM模型参数

        :Parameters:
            - obs: 观测序列O={o1, o2, ... oT}，值为观测集中的值
            - iters: 迭代次数
            - eps: 训练精度
        """
        for k in range(iters):
            # 训练
            self._map_b(obs)
            alpha =  self._forward(obs)
            beta = self._backward(obs)
            xi, gamma = self._baumwelch(obs, alpha, beta)
            model = self._estimate(obs, alpha, beta, xi, gamma)
            # 更新参数
            prob_old = lse(alpha[-1])
            self._updatemodel(model)
            prob_new = self.calc_prob(obs)
            # 检测收敛精度
            prod_d = abs(prob_old - prob_new)
            if verbose:
                print("Iter: {:3},"
                      " L(lambda|O) = {:.6e},"
                      " L(lambda_new|O) = {:.6e},"
                      " eps = {:.6f}"
                      .format(k, prob_old, prob_new, prod_d))
            if (prod_d < eps):
                break
