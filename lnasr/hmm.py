#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
隐马尔可夫模型（Hidden Markov Model, HMM）
"""

import numpy as np

class HMM:
    """
    HMM基本模型，形式化定义如下：

    - n: 隐状态数，状态集为Q={q1, q2, ... ,qN}
    - m: 观测集数量，观测集为V={v1, v2, ... ,vM}，一个实际的观测序列为O={o1, o2, ... oT}
    - A: 状态转移矩阵(NxN)，A[i, j]表示状态i转移到状态j
    - B: 发射概率(NxM)，B[i, ot]表示状态i生成观测ot概率（观测ot时状态为i的概率）
    - pi: 初始概率分布(Nx1)
    - obs: 一个观测度序列O(T)
    """

    def __init__(self, n:int, m:int,
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
        self.b = np.zeros((self.n, T), dtype=self.precision)
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
        # alpha = np.multiply(self.pi, self.b[:, 0])
        # p = np.copy(alpha)
        # for t in range(1, len(obs)):
        #     for j in range(self.n):
        #         alpha[j] = np.sum(np.multiply(p, self.A[:, j])) * self.b[j, t]
        #     p = np.copy(alpha)
        # return np.sum(alpha)
        T = len(obs)
        alpha = np.zeros((T, self.n), dtype=self.precision)
        # 初始状态
        alpha[0] = np.multiply(self.pi, self.b[:, 0])
        # 迭代
        for t in np.arange(1, T):
            for j in np.arange(self.n):
                alpha[t, j] = np.dot(alpha[t-1], self.A[:, j]) * self.b[j, t]
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
        beta = np.zeros((T, self.n), dtype=self.precision)
        # 初始状态
        beta[-1] = np.ones(self.n, dtype=self.precision)
        # 迭代
        for t in np.arange(T-2, -1, -1):
            for i in np.arange(self.n):
                beta[t, i] = np.sum(
                        np.multiply(np.multiply(beta[t+1], self.A[i, :]),
                            self.b[:, t+1]))
        return beta

    def _viterbi(self, obs:np.ndarray, remap:bool=False):
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
        v = np.zeros((T, self.n), dtype=self.precision)
        bt = np.zeros((T, self.n), dtype=np.uint32)
        # 初始状态
        v[0] = np.multiply(self.pi, self.b[:, 0])
        # 迭代
        for t in np.arange(1, T):
            for j in np.arange(self.n):
                val = np.multiply(v[t-1], self.A[:, j])
                v[t, j] = np.max(val) * self.b[j, t]
                bt[t, j] = np.argmax(val)
        # 反向追踪(backtrace)
        path = np.zeros(T, dtype=np.uint32)
        path[-1] = np.argmax(v[-1])
        for t in np.arange(T-2, -1, -1):
            path[t] = bt[t+1, path[t+1]]
        return (v, bt, path)

    def _baumwelch(self, obs:np.ndarray, alpha:np.ndarray, beta:np.ndarray, remap:bool=False):
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
        xi = np.zeros((T, self.n, self.n), dtype=self.precision)
        for t in np.arange(T-1):
            numer = np.multiply(alpha[t, :].reshape((self.n, 1)), self.A)
            numer = np.multiply(numer, self.b[:, t+1].reshape((1, self.n)))
            numer = np.multiply(numer, beta[t+1, :].reshape((1, self.n)))
            denom = np.sum(numer)
            xi[t] = numer / denom
        # gamma[t][i]: 对xi的j求和
        gamma = np.sum(xi, axis=2)
        return xi, gamma

    def _estimate(self, obs:np.ndarray, alpha, beta, xi, gamma):
        """计算Maximization模型新参数

        :Parameters:
            - obs: 观测序列O={o1, o2, ... oT}，值为观测集中的值
            - alpha: 向前传递因子(TxN)
            - xi: baum-welch参数(TxNxN)
            - gamma: baum-welch参数(TxN)

        :Returns: HMM模型参数(A,B,pi)
        """
        # M-Step: new A,B,pi
        denom = np.sum(gamma, axis=0)
        A = np.sum(xi, axis=0) / denom.reshape((self.n, 1))
        B = np.zeros((self.n, self.m), dtype=self.precision)
        for k in np.arange(self.m):
            B[:, k] = np.sum(gamma[k==obs, :], axis=0) / denom.reshape((1, self.n))
        pi = gamma[0]
        # New model
        model = {}
        model['A'] = A
        model['B'] = B
        model['pi'] = pi
        return model

    def _updatemodel(self, model):
        """更新模型参数"""
        self.A = model['A']
        self.B = model['B']
        self.pi = model['pi']

    def calc_prob(self, obs):
        """计算观测序列的似然度

        :Parameters:
            - obs: 观测序列O={o1, o2, ... oT}，值为观测集中的值
        """
        return np.sum(self._forward(obs, True)[-1])

    def decode(self, obs):
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
            prob_old = np.log(np.sum(alpha[-1]))
            self._updatemodel(model)
            prob_new = np.log(self.calc_prob(obs))
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
