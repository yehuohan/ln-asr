#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高斯混合模型（Gaussian Mixture Model, GMM）

 - log: 对数高斯概率以e为底（带_log的函数），但其中的均值、方差不用取对数。
"""

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.utils import lse
import numpy as np

def gaussian_distribution(x:np.ndarray, mu, sigma2)->np.ndarray:
    """单变量高斯分布

    :Parameters:
        - x: 自变量（Lx1）
        - mu: 均值
        - sigma2: 方差

    :Returns: 正态分布的值（Lx1）
    """
    return 1.0 / np.sqrt(2.0 * np.pi * sigma2) * np.exp(-(x - mu) * (x - mu) / (2.0 * sigma2))

def gaussian_distribution_log(x:np.ndarray, mu, sigma2)->np.ndarray:
    """单变量高斯分布（对数概率）"""
    return -0.5 * (np.log(2.0*np.pi) + np.log(sigma2) + (x-mu)*(x-mu)/sigma2)

def gaussian_multi_distribution(x:np.ndarray, mu:np.ndarray, Sigma:np.ndarray)->np.ndarray:
    """多变量高斯分布

    :Parameters:
        - x: 自变量（二维数组LxD，D为变量维度，D为1时即变成单变量高斯分布）
        - mu: 均值向量（1XD）
        - Sigma: 协方差矩阵（DxD）

    :Returns: 正态分布的值（Lx1）
    """
    L, D = x.shape
    det = np.linalg.det(Sigma)
    inv = np.asarray(np.mat(Sigma).I)
    xmu = x - mu
    r = 1.0 / (((2.0 * np.pi) ** (D / 2.0)) * ((det) ** 0.5))
    # 使用matmul实现for循环
    # val = np.empty(L, dtype=np.float32)
    # for k in np.arange(L):
    #     val[k] = r * np.exp(-0.5 * np.dot(xmu[k], inv).dot(xmu[k]))
    val = r * np.exp(-0.5 *
            # matmul(Lx1xD, LxDx1) -> LxDxD
            np.matmul(
                # matmul(Lx1xD, DxD) -> Lx1xD
                np.matmul(xmu.reshape(L, 1, D), inv).reshape(L, 1, D),
                xmu.reshape(L, D, 1)))
    return val.ravel()

def gaussian_multi_distribution_log(x:np.ndarray, mu:np.ndarray, Sigma:np.ndarray)->np.ndarray:
    """多变量高斯分布（对数概率）"""
    L, D = x.shape
    det = np.linalg.det(Sigma)
    inv = np.asarray(np.mat(Sigma).I)
    xmu = x - mu
    return -0.5 * (D*np.log(2.0*np.pi) + np.log(det) + \
            np.matmul(
                np.matmul(xmu.reshape(L, 1, D), inv).reshape(L, 1, D),
                xmu.reshape(L, D, 1)).ravel())

def gaussian_mixture_distribution(c:np.ndarray, x:np.ndarray, mu:np.ndarray, Sigma:np.ndarray):
    """混合高斯分布

    这里必须保证每个高斯分布的D相同。

    :Parameters:
        - c: 混合系数（Mx1）
        - x: 自变量（二维数组LxD，D为变量维度）
        - mu: 均值（MxD）
        - Sigma: 协方差矩阵（MxDxD)

    :Returns: 正态分布的值（Lx1）
    """
    M = c.shape[0]
    L, D = x.shape
    val = np.empty((M, L), dtype=np.float)
    for k in np.arange(M):
        val[k] = gaussian_multi_distribution(x, mu[k], Sigma[k])
    return np.dot(c.reshape(1, M), val)

def gaussian_mixture_distribution_log(c:np.ndarray, x:np.ndarray, mu:np.ndarray, Sigma:np.ndarray):
    """混合高斯分布（对数概率）

    混合系数需要取对数。
    """
    M = c.shape[0]
    L, D = x.shape
    val = np.empty((M, L), dtype=np.float)
    for k in np.arange(M):
        val[k] = gaussian_multi_distribution_log(x, mu[k], Sigma[k])
    return lse(c.reshape(M, 1) + val, axis=0)
