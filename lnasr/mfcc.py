#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
梅尔频率倒谱系数(Mel Frequency Cepstrum Coefficient)
"""

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.utils import split_frames
from lnasr.utils import create_hamming
import numpy as np
from scipy.fftpack import dct

class MFCC:
    """
    提取MFCC特征。
    """
    def __init__(self, fs = 16000,
            frame_T = 25e-3, frame_stride = 10e-3,
            hpf_alpha = 0.97,
            fft_N = 512,
            mfbank_num = 40, cepstrum_num = 12):
        """初始MFCC参数

        :Parameters:
            - fs: 采样频率Hz(frequency sample)
            - frame_T: 一帧时间长度，默认25ms
            - frame_stride: 默认帧移10ms（重叠15ms）
            - hpf_alpha: 一阶高通滤波参数
            - fft_N: FFT的宽度
            - mfbank_num: 滤波器组个数（Filter Bank）
            - cepstrum_num: 抽取倒谱值个数
        """
        self.fs = fs
        self.frame_T = frame_T
        self.frame_L = int(self.fs * self.frame_T)  # 一帧长度(frame length)，一帧数据点数
        self.frame_stride = frame_stride
        self.frame_step = int(self.fs * self.frame_stride)  # 帧的间隔，间隔数据点数
        self.hpf_alpha = hpf_alpha
        self.fft_N = fft_N
        self.fft_size = self.fft_N // 2 + 1     # FFT频谱的使用宽度
        self.mfbank_num = mfbank_num
        self.cepstrum_num = cepstrum_num

    def calc_high_pass_filter(self, signal:np.ndarray)->np.ndarray:
        """对信号进行高通滤

        公式如下：

        ..  math::
            y(t) = x(t) - \\alpha \cdot x(t-1)

        :Parameters:
            - signal 需要滤波的信号

        :Returns: 滤波后的信号
        """
        signal_hpf = np.append(signal[0], np.array(signal[1:] - self.hpf_alpha * signal[:-1]))
        return signal_hpf

    def calc_mel(self, hz:np.ndarray):
        """由Hertz频率计算Mel频率

        人耳对高频部分不太敏感，将频率转成Mel频率，当对Mel频率等距处理时，则原
        频率在高频部分的间距就很大。
        """
        return 2595 * np.log(1 + hz / 700)

    def calc_mel_inv(self, mel:np.ndarray):
        """由Mel频率计算Hertz频率"""
        return 700 * (np.exp(mel / 2595) - 1)

    def create_filter_bank(self)->np.ndarray:
        """生成Mel滤波器组(Filter Bank)。

        滤波器对应的Hertz频率范围为[0, fs/2]；
        在Mel频率范围内，滤波器组是等带宽的；
        每个滤波器收集来自给定频率范围内的能量；
        因为是用于对信号的频谱滤波，因而每个滤波器宽度与fft_size相同。
        """
        M = self.mfbank_num #滤波器组个数
        N = self.fft_N      #信号FFT的宽度
        # 计算每个滤波组（共M个）Hertz频率的起始和终止点
        hz_freq = self.calc_mel_inv(
                np.linspace(
                    self.calc_mel(0),
                    self.calc_mel(self.fs / 2),
                    M + 2))
        # 将hz_freq归一化到FFT频谱的坐标刻度上
        # hz = 0 + np.floor(((N - 0) / (self.fs - 0)) * (hz_freq - 0))
        hz = np.floor((N / self.fs) * hz_freq)
        # 生成M个滤波器
        mfbank = np.zeros((M, N // 2 + 1))
        for m in range(1, M + 1):
            lo =  int(hz[m - 1])
            mi =  int(hz[m])
            hi =  int(hz[m + 1])
            mfbank[m-1, lo:mi] = (np.arange(lo, mi) - lo) / (mi - lo)
            mfbank[m-1, mi:hi] = (hi - np.arange(mi, hi)) / (hi - mi)

        return mfbank

    def calc_mfcc(self, signal:np.ndarray):
        """计算MFCC特征

        - 预加重
            使用一阶高通滤波，增加高频段的能量；
        - 分帧
            将信号分成若干个长度为frame_L个点的帧，一帧代表的时间长度为frame_T秒；
        一帧的时间长度为frame_T=25ms，对应的数据点数为frame_L=400。
        分帧的数据为一个数组，数组元素是长度为400的帧数据。
        - 加窗
            窗的长度等同于帧长度：
        - STFT
            频谱的只用正频率，范围为[0, fs/2]，对应频域坐标范围为[0, fft_size]；
        - Mel Filter Bank
            生成滤波器组，用于对信号频谱滤波，因滤波器宽度同为fft_size个点；
        - 倒谱
            倒谱是声谱对数的声谱，即对声音能量谱的对数做DFT；

        :Parameters:
            - signal 语音信号数据

        :Returns: 返回MFCC特征
        """
        # 预加重
        signal_hpf = self.calc_high_pass_filter(signal)

        # 分帧[frames x frame_L]
        frames_signal = split_frames(signal_hpf, self.frame_L, self.frame_step)

        # 加窗
        frames_signal *= create_hamming(self.frame_L)

        # STFT[frames x fft_size]
        frames_magnitude = np.abs(np.fft.rfft(frames_signal, self.fft_N))
        frames_power = ((1.0 / self.fft_N) * (frames_magnitude ** 2))   # 帕塞瓦尔定理

        # Mel Filter Bank[mfbank_num x fft_size]
        mfbank = self.create_filter_bank()

        # 倒谱[frames_num x mfbank_num]
        frames_cepstrum = np.dot(frames_power, mfbank.T)
        frames_cepstrum = np.where(frames_cepstrum == 0, np.finfo(float).eps, frames_cepstrum)  # 用无限接近零的浮点数值代替0
        frames_cepstrum = 20 * np.log10(frames_cepstrum)    # dB
        frames_cepstrum -= (np.mean(frames_cepstrum, axis=0) + 1e-8)
        frames_cepstrum = dct(frames_cepstrum, type=2, axis=1, norm='ortho')

        # features[frames_num x 12]: 前12个倒谱系数
        features = frames_cepstrum[:, 1:(1+self.cepstrum_num)]
        features -= (np.mean(features, axis=0) + 1e-8)
        # features[frames x 13]: 能量特征（取对数）
        features = np.concatenate(
                (features, np.log(np.sum(frames_power, axis=1).reshape((-1, 1)))),
                axis=1)
        # features[frames_num x 26]: delta特征
        features = np.concatenate(
                (features, np.concatenate(
                    (features[1].reshape((1, -1)), features[1:] - features[:-1]),
                    axis=0)),
                axis=1)
        # features[frames_num x 39]: 双delta特征
        features = np.concatenate(
                (features, np.concatenate(
                    (features[1, 13:26].reshape((1, -1)), features[1:, 13:26] - features[:-1, 13:26]),
                    axis=0)),
                axis=1)

        # return mfcc features
        return (frames_power, frames_cepstrum, features)
