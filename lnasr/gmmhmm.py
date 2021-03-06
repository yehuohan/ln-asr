#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高斯混合模型-隐马尔可夫模型(GMM-HMM)
"""

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.utils import lse
from lnasr.gmm import *
from lnasr.hmm import HMM
import numpy as np
import h5py

class GMMHMM(HMM):
    """
    基于GMM的HMM，属性如下：

    - n: 隐状态数，状态集为Q={q1, q2, ... ,qN}
    - m: 高斯分布个数，相当于观测集合元素个数
    - d: 特征向量的维度（例如：MFCC特征为D=39维）
    - A: 状态转移矩阵(NxN)，A[i, j]表示状态i转移到状态j
    - pi: 初始概率分布(Nx1)
    - w: 混合系数(NxM)
    - mu: 均值矢量(NxMxD)，
    - Sigma: 协方差矩阵(NxM x DXD)
    - obs: 一个观测度序列O(TxD)

    A, pi, w的概率值使用log概率，对数底为e。
    """

    def __init__(self, n:int=1, m:int=1, d:int=1,
            A:np.matrix=None, pi:np.ndarray=None,
            w:np.ndarray=None, mu:np.ndarray=None, si:np.ndarray=None,
            precision=np.double):
        """初始化GMMHMM参数"""
        super().__init__(n, m, A, None, pi, precision)
        self.d = d
        self.w = w
        self.mu = mu
        self.si = si
        self.min_std = 0.01

    def _map_b(self, obs:np.ndarray):
        """利用GMM计算发射概率

        b与GMM计算的映射关系：

        - b: (NxT), b[j, t]表示从状态j生成ot的概率
        - bm: (NxMxT), b[j, m, t]表示从状态j生成ot的概率的第m个混合分量

        :Parameters:
            - obs: 观测序列
        """
        T = obs.shape[0]
        self.b = np.empty((self.n, T), dtype=self.precision)
        self.bm = np.empty((self.n, self.m, T), dtype=self.precision)
        # 可以直接使用gaussian_mixture_distribution计算，但没法获取混合分量
        # for j in np.arange(self.n):
        #     self.b[j] = gaussian_mixture_distribution_log(
        #             self.w[j], obs, self.mu[j], self.si[j])
        for j in np.arange(self.n):
            for m in np.arange(self.m):
                self.bm[j, m] = gaussian_multi_distribution_log(obs, self.mu[j, m], self.si[j, m])
            self.b[j] = lse(self.w[j].reshape(self.m, 1) + self.bm[j], axis=0)

    def _estimate(self, obs:np.ndarray, alpha, beta, xi, gamma):
        """计算Maximization模型新参数

        :Parameters:
            - obs: 观测序列O[TxD]={o1, o2, ... oT}，值为观测集中的值
            - alpha: 向前传递因子(TxN)
            - xi: baum-welch参数(TxNxN)
            - gamma: baum-welch参数(TxN)

        :Returns: HMM模型参数(A,pi,mu,si,w)
        """
        T = obs.shape[0]
        # M-Step: new A,pi,mu,si,w
        # A[NxN]
        A = lse(xi, axis=0) - lse(gamma, axis=0).reshape((self.n, 1))

        # pi[N]
        pi = gamma[0]

        # xi_mix[TxNxM]
        xi_mix = np.empty((T, self.n, self.m), dtype=self.precision)
        for t in np.arange(T):
            """
            # 用2层for循环，便于理解数组乘法代替for循环
            for j in np.arange(self.n):
                xi_mix[t, j, :] = (alpha[t, j] + beta[t, j] + self.w[j, :] + self.bm[j, :, t])
                xi_mix[t, j, :] -= lse(alpha[t] + beta[t])
                xi_mix[t, j, :] -= lse(self.w[j] + self.bm[j, :, t])
            """
            xi_mix[t] = (alpha[t] + beta[t]).reshape(self.n, 1) + self.w + self.bm[:, :, t]
            xi_mix[t] -= lse(alpha[t]+ beta[t])
            xi_mix[t] -= lse(self.w + self.bm[:, :, t], axis=1).reshape(self.n, 1)

        # w[NxM]
        w = lse(xi_mix, axis=0) - lse(xi_mix, axis=(0, 2)).reshape(self.n, 1)

        # xi_mix取exp，用于计算非log值的均值和方差
        xi_mix = np.exp(xi_mix)

        # mu[NxMxD]
        """
        # 用2层for循环
        mu = np.zeros((self.n, self.m, self.d), dtype=self.precision)
        for i in np.arange(self.n):
            for m in np.arange(self.m):
                mu[i, m] = np.dot(xi_mix[:, i, m].reshape(1, T), obs)
                mu[i, m] /= np.sum(xi_mix[:, i, m])
        """
        mu = np.dot(
                # dot(TxNxM -> NxMxT, TxD) -> NxMxD
                np.swapaxes(np.swapaxes(xi_mix, 0, 1), 1, 2),
                obs) / np.sum(xi_mix, axis=0).reshape(self.n, self.m, 1)

        # Sigma[NxMxDxD]
        """
        # 用2层for循坏
        si = np.zeros((self.n, self.m, self.d, self.d), dtype=self.precision)
        for i in np.arange(self.n):
            for m in np.arange(self.m):
                dt = obs - self.mu[i, m]  # TxD
                si[i, m] = np.sum(
                        xi_mix[:, i, m].reshape(T, 1, 1) * np.matmul(
                            # matmul(TxDx1, Tx1xD) -> TxDxD
                            dt.reshape(T, self.d, 1),
                            dt.reshape(T, 1, self.d)),
                        axis=0)
                si[i, m] /= np.sum(xi_mix[:, i, m])
        """
        # dt = 1x1xTxD - NxMx1xD = NxMxTxD
        dt = obs.reshape(1, 1, T, self.d) - self.mu.reshape(self.n, self.m, 1, self.d)
        si = np.sum(
                # TxNxM -> NxMxT -> NxMxTx1x1
                np.swapaxes(np.swapaxes(xi_mix, 0, 1), 1, 2).reshape(self.n, self.m, T, 1, 1) * \
                np.matmul(
                    # matmul(NxMxTxDx1, NxMxTx1xD) -> NxMxTxDxD
                    dt.reshape(self.n, self.m, T, self.d, 1),
                    dt.reshape(self.n, self.m, T, 1, self.d)),
                axis=2) / np.sum(xi_mix, axis=0).reshape(self.n, self.m, 1, 1)
        si[:, :] += np.matrix(self.min_std * np.eye((self.d), dtype=self.precision))

        # New model
        model = {}
        model['A'] = A
        model['pi'] = pi
        model['w'] = w
        model['mu'] = mu
        model['si'] = si
        return model

    def _updatemodel(self, model):
        """更新模型参数"""
        self.A = model['A']
        self.pi = model['pi']
        self.mu = model['mu']
        self.si = model['si']
        self.w = model['w']

    def reset(self, init_type):
        """初始化训练参数

        :Returns:
            - init_type: 初始化类型，默认全部初始化为零
        """
        if init_type == 'uniform':
            self.A = np.log(np.ones((self.n, self.n), dtype=self.precision) * (1.0 / self.n))
            self.pi = np.log(np.ones(self.n, dtype=self.precision) * (1.0 / self.n))
            self.w = np.log(np.ones((self.n, self.m), dtype=self.precision) * (1.0 / self.m))
            self.mu = np.zeros((self.n, self.m, self.d), dtype=self.precision)
            self.si = np.zeros((self.n, self.m, self.d, self.d), dtype=self.precision)
            self.si[:, :] = np.eye(self.d, dtype=self.precision)
        elif init_type == 'random':
            a = np.random.random_sample((self.n, self.n))
            pi = np.random.random_sample((self.n))
            w = np.random.random_sample((self.n, self.m))
            self.A = np.log(a / a.sum(axis=1)[:, np.newaxis])
            self.pi = np.log(pi / np.sum(pi))
            self.w = np.log(w / w.sum(axis=1)[:, np.newaxis])
            self.mu = np.array(0.6 * np.random.random_sample((self.n, self.m, self.d)) - 0.3, dtype=self.precision)
            self.si = np.zeros((self.n, self.m, self.d, self.d), dtype=self.precision)
            self.si[:, :] = np.eye((self.d), dtype=self.precision)

    def save(self, filename):
        """保存模型参数"""
        f = h5py.File(filename, 'w')
        f.create_dataset('A', data=self.A)
        f.create_dataset('pi', data=self.pi)
        f.create_dataset('w', data=self.w)
        f.create_dataset('mu', data=self.mu)
        f.create_dataset('si', data=self.si)
        f.close()

    def load(self, filename):
        """加载模型参数"""
        f = h5py.File(filename, 'r')
        self.A = f['A']
        self.pi = f['pi']
        self.w = f['w']
        self.mu = f['mu']
        self.si = f['si']
        self.n, self.m, self.d = self.mu.shape
