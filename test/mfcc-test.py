#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.mfcc import MFCC
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from scipy.fftpack import dct
import numpy as np

mpl.rcParams['font.family'] = 'Consolas'
mpl.rcParams['font.size'] = 11

def on_key(event:mpl.backend_bases.KeyEvent):
    if event.key == 'escape':
        plt.close()

def get_voice_raw(filepath)->np.ndarray:
    """读取语音数据(16KHz, 16bit)"""
    with open(filepath, 'rb') as fp:
        byte = fp.read()
        size = fp.tell() // 2
        vr = np.zeros((size), dtype=np.int16)
        for k in range(size):
            vr[k] = (byte[2*k]) | (byte[2*k+1]<<8)
    return vr

#%% 读取语音数据
vr = get_voice_raw("data.raw").astype(dtype=np.float, copy=False)
vr_fft = np.fft.rfft(vr)
fig = plt.figure('Voice data')
fig.canvas.mpl_connect('key_press_event', on_key)
ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
ax.set_xlabel('Time')
ax.set_xticks(np.linspace(0, len(vr), 5))
ax.set_xticklabels((str(x) for x in np.linspace(0, len(vr)/16000.0, 5)))
ax.plot(vr)
# fig = plt.figure('Voice data fft')
# fig.canvas.mpl_connect('key_press_event', on_key)
# ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
# ax.set_xlabel('Hz')
# ax.plot(np.linspace(1, 16000//2, num=len(vr_fft)), vr_fft)

#%% MFCC计算
mfcc = MFCC(
        fs=16000,
        frame_T=25e-3,
        frame_stride=10e-3,
        hpf_alpha=0.97,
        fft_N=512,
        mfbank_num=40,
        cepstrum_num=12
        )

frames_power, frames_cepstrum, features_mfcc = mfcc.calc_mfcc(vr)

# 预加重
vr_hpf = mfcc.calc_high_pass_filter(vr)
vr_hpf_fft = np.fft.rfft(vr_hpf)
fig = plt.figure('Voice data hpf')
fig.canvas.mpl_connect('key_press_event', on_key)
ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
ax.plot(vr_hpf)
ax.set_xlabel('Time')
ax.set_xticks(np.linspace(0, len(vr), 5))
ax.set_xticklabels((str(x) for x in np.linspace(0, len(vr)/16000.0, 5)))
# fig = plt.figure('Voice data hpf fft')
# fig.canvas.mpl_connect('key_press_event', on_key)
# ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
# ax.set_xlabel('Hz')
# ax.plot(np.linspace(1, 16000//2, num=len(vr_hpf_fft)), vr_hpf_fft)

# 分帧
vr_frames = mfcc.split_frames(vr_hpf)
fig = plt.figure('Voice frames')
fig.canvas.mpl_connect('key_press_event', on_key)
ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
ax.imshow(vr_frames, cmap='flag')

# 加窗
vr_win = vr_frames * mfcc.create_hamming()
fig = plt.figure('Voice window')
fig.canvas.mpl_connect('key_press_event', on_key)
ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
ax.imshow(vr_win, cmap='flag')

# 加窗前后的数据变化
fig = plt.figure('Voice frame and window')
fig.canvas.mpl_connect('key_press_event', on_key)
ax = fig.add_subplot(2, 1, 1)
ax.plot(vr_frames[100], label='frame[100]')
ax.legend(loc='lower right')
ax = fig.add_subplot(2, 1, 2)
ax.plot(vr_win[100], label='frame[100] in window')
ax.legend(loc='lower right')

# 短时傅里叶变换
fig = plt.figure('Voice power')
fig.canvas.mpl_connect('key_press_event', on_key)
ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
ax.imshow(frames_power, cmap='flag')
ax.set_xticks(np.linspace(0, frames_power.shape[1], 5))
ax.set_xticklabels((str(int(x)) for x in np.linspace(0, 16000//2, 5)))

# Mel Filter Bank
mfbank = mfcc.create_filter_bank()
fig = plt.figure('Mel Filter Bank')
fig.canvas.mpl_connect('key_press_event', on_key)
ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
for k in range(mfcc.mfbank_num):
    ax.plot(np.linspace(1, mfcc.fs//2, num=mfcc.fft_size), mfbank[k, :])
# ax.plot(np.linspace(1, mfcc.fs//2, num=mfcc.fft_size), mfbank[1, :], 'r')

# 倒谱
fig = plt.figure('Cepstrum')
fig.canvas.mpl_connect('key_press_event', on_key)
ax = fig.add_axes((0.15, 0.1, 0.75, 0.85))
ax.imshow(frames_cepstrum, cmap='jet', aspect='auto', interpolation='nearest')
ax.set_xticks(np.linspace(0, 40, 5))
ax.set_xticklabels((str(int(x)) for x in np.linspace(0, 40, 5)))

# MFCC特征
fig = plt.figure('Mfcc features')
fig.canvas.mpl_connect('key_press_event', on_key)
ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
ax.imshow(features_mfcc, cmap='jet', aspect='auto', interpolation='nearest')
ax.set_xticks(np.linspace(0, 39, 4))
ax.set_xticklabels((str(int(x)) for x in np.linspace(0, 39, 4)))
plt.show()
