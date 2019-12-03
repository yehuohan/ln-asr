#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voice Activity Detection.

:VadLtsd:

    - `Efficient voice activity detection algorithms using long-term speech information <https://www.sciencedirect.com/science/article/abs/pii/S0167639303001201>`_
    - https://github.com/isrish/VAD-LTSD
"""

import sys,os
sys.path.append(os.getcwd() + '/../../')

from lnasr.utils import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pysnooper

def on_key(event:mpl.backend_bases.KeyEvent):
    if event.key == 'escape':
        plt.close()

class VadLtsd:
    """静音检测"""

    def __init__(self, freq=16000,
            winsize=2048, stepsize=1024,
            order=6, threshold=-6, alpha=None):
        """初始化Vad

        :Parameters:
            - freq: 音频频率，Hz
            - winsize: 一帧长度，也是FFT的宽度
            - stepsize: 分帧时的帧移长度
            - order: 计算ltse和ltsd阶数
        """
        self.freq = freq 
        self.winsize = winsize
        self.stepsize = stepsize
        self.order = order
        self.threshold = threshold
        self.alpha = alpha
        self.fft_size = self.winsize // 2 + 1       # 频率使用宽度

    # @pysnooper.snoop()
    def detect(self, data:np.ndarray) -> np.ndarray:
        """对一段语音进行检测

        :Parameters:
            - data: 待检测语音序列

        :Returns: 每帧语音的LTSD计算结果
        """
        pad = np.append(np.zeros((self.stepsize)), data)    # 模拟matlab buffer效果
        # 分帧[frames_num x winsize]
        frames = split_frames(pad, self.winsize, self.stepsize)
        frames_num = frames.shape[0]
        # 加窗(hamming)
        frames *= create_hamming(self.winsize)
        # 使用FFT计算幅度频谱[frames_num x fft_size]
        frames_amplitude = np.empty((frames_num, self.fft_size))
        frames_amplitude = np.abs(np.fft.rfft(frames, self.winsize))
        # 基于前2帧计算noise
        noise = np.average(frames_amplitude[0:2], axis=0) ** 2
        # 计算LTSE(Long-Term Spectral Envelope)
        ltse = np.zeros((frames_num, self.fft_size))
        for k in np.arange(self.order,frames_num - self.order):
            ltse[k] = frames_amplitude[k-self.order:k+self.order].max(axis=0)
        # 计算LTSD(Long-Term Spectral Divergence)
        ltsd = np.zeros(frames_num, dtype=np.double)
        for k in np.arange(self.order,frames_num - self.order):
            ltsd[k] = np.sum(ltse[k] ** 2 / noise)
            ltsd[k] = 10 * np.log10(ltsd[k] / self.winsize)
            if None != self.alpha and ltsd[k] < self.threshold:
                noise = self.alpha*noise + (1-self.alpha)*(np.sum(ltse[k]) / self.winsize)
        return ltsd

if __name__ == "__main__":
    wd, fs = read_wave("data.wav")
    wd = np.array(wd / (65536.0/2), dtype=np.double)
    vad = VadLtsd(
            freq=fs,
            winsize=512,
            stepsize=256,
            order=4,
            threshold=-6,
            alpha=0.4)
    ltsd = vad.detect(wd)
    # 修改order可以改LTSE的包络宽度（order增大，会提前检测到Activity位置，也会延后Non-Activity位置）
    res_points = np.arange(ltsd.shape[0]) * vad.stepsize
    res_nor = ltsd / np.max(ltsd)
    res_bin = (res_nor > 0.25) * np.max(wd)
    plt.figure('Plot')
    plt.connect('key_press_event', on_key)
    plt.plot(wd, linewidth=0.5)
    plt.plot(res_points, res_nor, 'g')
    plt.plot(res_points, res_bin, 'r')
    plt.show()
