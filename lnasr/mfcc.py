
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mel Frequency Cepstrum Coefficient.

Ref:
 - https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
"""

import numpy as np
from scipy.fftpack import dct

class MFCC:
    """
    提取MFCC特征。
    """
    def __init__(self, **kwargs):
        self.fs = 16000                 # 采样频率Hz(frequency sample)
        self.frame_T = 25e-3            # 一帧25ms
        self.frame_L = int(self.fs * self.frame_T)      # 一帧长度(frame length)，一帧数据点数
        self.frame_stride = 10e-3       # 帧移10ms（重叠15ms）
        self.frame_step = int(self.fs * self.frame_stride)      # 帧的间隔，间隔数据点数
        self.hpf_alpha = 0.97           # 一阶高通滤波参数
        self.fft_N = 512                # FFT的宽度
        self.fft_size = self.fft_N // 2 + 1     # FFT频谱的使用宽度
        self.mfbank_num = 40            # 滤波器组个数（Filter Bank）
        self.cepstrum_num = 12          # 抽取倒谱值个数
        self.init_mfcc(**kwargs)

    def init_mfcc(self, **kwargs):
        """MFCC参数初始化"""
        if 'fs' in kwargs:
            self.fs = kwargs['fs']
        if 'T' in kwargs:
            self.frame_T = kwargs['T']
        if 'stride' in kwargs:
            self.frame_stride = kwargs['stride']
        self.frame_L = int(self.fs * self.frame_T)
        self.frame_step = int(self.fs * self.frame_stride)

        if 'alpha' in kwargs:
            self.hpf_alpha = kwargs['alpha']
        if 'N' in kwargs:
            self.fft_N = kwargs['N']
            self.fft_size = self.fft_N // 2 + 1
        if 'm_num' in kwargs:
            self.mfbank_num = kwargs['m_num']
        if 'c_num' in kwargs:
            self.cepstrum_num = kwargs['c_num']

    def calc_high_pass_filter(self, signal:np.ndarray, alpha)->np.ndarray:
        """对信号进行高通滤
        公式如下：
        y(t) = x(t) - a * x(t-1)

        :Parameters:
        - signal 需要滤波的信号
        - alpha 一阶高通滤波参数

        :Returns:
        滤波后的信号
        """
        signal_hpf = np.append(signal[0], np.array(signal[1:] - alpha * signal[:-1]))
        return signal_hpf

    def __split_frames(self, signal:np.ndarray)->np.ndarray:
        """对语音信号分帧"""
        signal_length = len(signal)     # 信号长度
        frames_num = int(np.ceil(np.abs(signal_length - self.frame_L) / self.frame_step)) # 帧的数量

        # 未尾帧填充0，保证每帧长度相同
        signal_pad_length = frames_num * self.frame_step + self.frame_L
        if signal_pad_length > signal_length:   # 最后一帧用0填充至frame_L的长度
            signal_pad = np.append(signal, np.zeros((signal_pad_length - signal_length)))
        else:
            signal_pad = signal

        # 用frames_signal保存信号每帧数据
        indices = np.tile(np.arange(0, self.frame_L), (frames_num, 1)) + \
            np.tile(np.arange(0, frames_num * self.frame_step, self.frame_step), (self.frame_L, 1)).T
        frames_signal = signal_pad[indices.astype(np.int32, copy=False)]
        return frames_signal

    def create_hamming(self, N)->np.ndarray:
        """生成宽度为N个点的汉明窗
        公式如下（N为窗长度）：
            w(n) = 0.54 - 0.46 * cos(2 * pi * n / (N - 1))
        """
        return (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1)))

    def calc_mel(self, hz:np.ndarray):
        """由Hertz频率计算Mel频率
        人耳对高频部分不太敏感，将频率转成Mel频率，当对Mel频率等距处理时，则原
        频率在高频部分的间距就很大。
        """
        return 2595 * np.log(1 + hz / 700)

    def calc_mel_inv(self, mel:np.ndarray):
        """由Mel频率计算Hertz频率"""
        return 700 * (np.exp(mel / 2595) - 1)

    def create_filter_bank(self, M, N)->np.ndarray:
        """生成Mel滤波器组(Filter Bank)。
        滤波器对应的Hertz频率范围为[0, fs/2]；
        在Mel频率范围内，滤波器组是等带宽的；
        每个滤波器收集来自给定频率范围内的能量；
        因为是用于对信号的频谱滤波，因而每个滤波器宽度与fft_size相同。

        :Parameters:
        - M 滤波器组个数
        - N 信号FFT的宽度
        """
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

        :Returns:
        返回MFCC特征
        """
        # 预加重
        signal_hpf = self.calc_high_pass_filter(signal, self.hpf_alpha)

        # 分帧
        frames_signal = self.__split_frames(signal_hpf)

        # 加窗
        frames_signal *= self.create_hamming(self.frame_L)

        # STFT
        frames_magnitude = np.abs(np.fft.rfft(frames_signal, self.fft_N))
        frames_power = ((1.0 / self.fft_N) * (frames_magnitude ** 2))

        # Mel Filter Bank
        mfbank = self.create_filter_bank(self.mfbank_num, self.fft_N)

        # 倒谱
        frames_cepstrum = np.dot(frames_power, mfbank.T)
        frames_cepstrum = np.where(frames_cepstrum == 0, np.finfo(float).eps, frames_cepstrum)  # 用无限接近零的浮点数值代替0
        frames_cepstrum = 20 * np.log10(frames_cepstrum)    # dB
        frames_cepstrum -= (np.mean(frames_cepstrum, axis=0) + 1e-8)

        features_mfcc = dct(frames_cepstrum, type=2, axis=1, norm='ortho')[:, 1:(1+self.cepstrum_num)]
        features_mfcc -= (np.mean(features_mfcc, axis=0) + 1e-8)

        # return features_mfcc
        return (frames_signal, frames_magnitude, frames_power, frames_cepstrum, features_mfcc)
