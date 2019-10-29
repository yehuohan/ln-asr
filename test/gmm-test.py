#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.gmm import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import numpy as np

mpl.rcParams['font.family'] = 'Consolas'
mpl.rcParams['font.size'] = 11

def on_key(event:mpl.backend_bases.KeyEvent):
    if event.key == 'escape':
        plt.close()

#%% 单变量高斯分布
mu = 0
sigma2 = 4
L = 1000
x_points = np.linspace(-10, 10, L)
f_points = gaussian_distribution(x_points, mu, sigma2)
# print(np.log(f_points[:100]))
# print(gaussian_distribution_log(x_points, mu, sigma2)[:100])
# f_points = gaussian_multi_distribution(
#         np.expand_dims(x_points, axis=1),
#         np.array([0]).reshape(1, 1),
#         np.array([4]).reshape(1, 1))    # 单维也属于用多维高斯分布
print("面积： ", np.abs(np.sum((x_points[:-1] - x_points[1:]) * f_points[:-1])))
plt.figure('Gaussian single variable')
plt.connect('key_press_event', on_key)
plt.axes((0.1, 0.1, 0.8, 0.8))
plt.plot(x_points, f_points)
plt.show()


#%% 多变量高斯分布
mu = np.array([0, 0]),
Sigma = np.array([[1.0, 0.3],
                  [0.2, 1.0]], dtype=np.float)
L = 1000
r = np.linspace(-3, 3, L, dtype=np.float)
mesh = np.meshgrid(r, r)
x_points = np.empty((L*L, 2), dtype=np.float)
x_points[:, 0] = mesh[0].ravel()
x_points[:, 1] = mesh[1].ravel()
f_points = gaussian_multi_distribution(x_points, mu, Sigma)
f_points = f_points.reshape((L, L))
# print(np.log(f_points[:10, :10]))
# print(gaussian_multi_distribution_log(x_points, mu, Sigma).reshape((L, L))[:10, :10])
plt.figure('Gaussian multi variable')
plt.connect('key_press_event', on_key)
ax = plt.subplot(121, projection='3d')
ax.plot_surface(mesh[0], mesh[1], f_points, shade=True, cmap='jet')
ax = plt.subplot(122)
ax.contourf(mesh[0], mesh[1], f_points, cmap='rainbow')
ax.contour(mesh[0], mesh[1], f_points, colors='black')
plt.show()


#%% 高斯混合模型
M = 3
L = 100
c = np.array([0.2, 0.3, 0.5], dtype=np.float)
x = np.empty((L*L, 2), dtype=np.float)
r = np.linspace(-6, 6, L, dtype=np.float)
mesh = np.meshgrid(r, r)
x[:, 0] = mesh[0].ravel()
x[:, 1] = mesh[1].ravel()
mu = np.empty((M, 2), dtype=np.float)
mu[0] = np.array([0, 0], dtype=np.float)
mu[1] = np.array([-3, -3], dtype=np.float)
mu[2] = np.array([3, 3], dtype=np.float)
Sigma = np.empty((M, 2, 2), dtype=np.float)
Sigma[0] = np.array([[1.0, 0.3],
                     [0.2, 1.0]], dtype=np.float)
Sigma[1] = np.array([[1.0, 0.3],
                     [0.2, 1.0]], dtype=np.float)
Sigma[2] = np.array([[1.0, 0.3],
                     [0.2, 1.0]], dtype=np.float)
f = gaussian_mixture_distribution(c, x, mu, Sigma).reshape(L, L)
plt.figure('Gaussian mixture model')
plt.connect('key_press_event', on_key)
ax = plt.subplot(121, projection='3d')
ax.plot_surface(mesh[0], mesh[1], f, shade=True, cmap='jet')
ax = plt.subplot(122)
ax.contourf(mesh[0], mesh[1], f, cmap='rainbow')
ax.contour(mesh[0], mesh[1], f, colors='black')
plt.show()
