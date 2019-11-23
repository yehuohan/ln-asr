#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
一些共用函数
"""

import numpy as np

def split_frames(signal:np.ndarray, L, S):
    """信号分帧

    最后不足一帧时，补零处理。

    :Parameters:
        - signal: 数据序序
        - L: 一帧的长度
        - S: 帧移长度（帧之间的间隔，应小于L）

    :Returns:
        - 分帧的语音数据[N x L]
    """
    signal_length = len(signal)
    N = int(np.ceil(np.abs(signal_length - (L - S)) / S)) # 帧的数量，保证至少有1帧
    # 未尾帧填充0，保证每帧长度相同
    pad_length = N * S + (L - S)
    if pad_length > signal_length:   # 最后一帧用0填充至frame_L的长度
        pad = np.append(signal, np.zeros((pad_length - signal_length)))
    else:
        pad = signal
    # 用frames保存信号每帧数据
    indices = np.tile(np.arange(0, L), (N, 1)) + \
              np.tile(np.arange(0, N * S, S), (L, 1)).T
    frames = pad[indices.astype(np.int32, copy=False)]
    return frames

def create_hamming(N:int)->np.ndarray:
    """生成宽度为N汉明窗

    ..  math::
        w(n) = 0.54 - 0.46 \cdot cos(2 \cdot \\pi \cdot \cfrac{n}{N - 1})
    """
    return (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1)))

if __name__ == "__main__":
    a = np.arange(30)
    print(split_frames(a, 3, 2))
