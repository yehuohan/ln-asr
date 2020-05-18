#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
一些常用函数
"""

import numpy as np
import pyaudio
import wave

Punctuation_Ascii_Set = {
    '.',
    ',',
    '?',
    '!',
    '"',
    "'",
    ':',
    ';',
    '(',
    ')',
    '[',
    ']',
    '{',
    '}'
    }

Punctuation_Unicode_Set = {
    '。',
    '，',
    '？',
    '！',
    '：',
    '；',
    '、',
    '‘',
    '’',
    '“',
    '”',
    '—',
    '《',
    '》',
    '（',
    '）',
    '【',
    '】'
    }

def recording(time, pcmfile=None, wavefile=None)->np.ndarray:
    """获取录音PCM数据

    PCM为单通道，16KHz，16bit，小端格式

    :Parameters:
        - time: 录音时间，单位为秒，可为浮点数
        - pcmfile: 保存PCM文件
        - waveilfe: 保存wave文件

    :Returns: 返回录音数据
    """
    pa = pyaudio.PyAudio()
    stream = pa.open(rate=16000, channels=1, format=pyaudio.paInt16,
            input=True, frames_per_buffer=160)
    print("start recording...")
    frames = []
    for k in range(int(16000 / 160 * time)):
        # 共16000*time个16bit采样点，每次读取160个16bit采样点
        data = stream.read(160)
        frames.append(data)
    print("stop recording")
    stream.stop_stream()
    stream.close()
    pa.terminate()
    data = np.frombuffer(b''.join(frames), dtype=np.int16)
    # 保存音频
    if pcmfile != None:
        data.tofile(pcmfile)
    if wavefile != None:
        fp = wave.open(wavefile, 'wb')
        fp.setnchannels(1)
        fp.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        fp.setframerate(16000)
        fp.writeframes(data)
        fp.close()
    # 转换成ndarray格式
    return data

def read_wave(filename:str)->np.ndarray:
    """获取wave音频数据和频率

    wave音频为单通道，16bit，小端格式

    :Returns: 返回音频序列和频率
    """
    fp = wave.open(filename, 'rb')
    wparams = fp.getparams()
    data = fp.readframes(wparams.nframes * wparams.sampwidth)
    fp.close()
    wdata = np.frombuffer(data, dtype=np.int16)
    return (wdata, wparams.framerate)

def read_pcm(filename:str)->np.ndarray:
    """读取PCM语音数据(单通道，16bit，小端)"""
    with open(filename, 'rb') as fp:
        data = fp.read()
        return np.frombuffer(data, dtype=np.int16)

def split_frames(signal:np.ndarray, L, S):
    """信号分帧

    不足一帧时，补零处理。

    :Parameters:
        - signal: 数据序序
        - L: 一帧的长度
        - S: 帧移长度（帧之间的间隔，应小于L）

    :Returns: 分帧的语音数据[N x L]
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

def lse(seq:np.ndarray, axis=None)->np.ndarray:
    """log-sum-exp

    计算log(sum(e**si))，即log(e**s1 + e**s2 + ...)，因为有sum，故会降低一个维度。
    axis为None时，取所有元素；
    axis为int时，对维度遍历；
    axis为tuple，对tuple中维度依次遍历，注意：tuple的维度必须升序排列。

    ::

        设 y = x + d
        log(e**x + e**y) = log(e**x * (1 + e**d))
                         = x + log(1 + e**d)

    :Parameters:
        - axis: 计算维度
    """
    if axis == None:
        return lse(seq.ravel(), 0)
    elif isinstance(axis, tuple):
        if len(axis) == 0:
            return seq
        elif len(axis) == 1:
            return lse(seq, axis[0])
        elif len(axis) > 1:
            return lse(lse(seq, axis[-1]), axis[:-1])
    elif isinstance(axis, int):
        if seq.shape[axis] == 0:
            return np.full(tuple(filter(lambda d:d!=0, seq.shape)), -np.inf)
        elif seq.shape[axis] == 1:
            return seq.take(0, axis)
        elif seq.shape[axis] == 2:
            return np.logaddexp(seq.take(0, axis), seq.take(1, axis))
        else:
            # 对于长度列，递归计算可能超过最大递归次数
            # return np.logaddexp(seq.take(0, axis), lse(seq.take(np.arange(1, seq.shape[axis]), axis), axis))
            res = seq.take(0, axis)
            for k in np.arange(1, seq.shape[axis]):
                res = np.logaddexp(res, seq.take(k, axis))
            return res

def lse2(seq:np.ndarray, axis=None)->np.ndarray:
    """log-sum-exp2"""
    if axis == None:
        return lse2(seq.ravel(), 0)
    elif isinstance(axis, tuple):
        if len(axis) == 0:
            return seq
        elif len(axis) == 1:
            return lse2(seq, axis[0])
        elif len(axis) > 1:
            return lse2(lse2(seq, axis[-1]), axis[:-1])
    elif isinstance(axis, int):
        if seq.shape[axis] == 0:
            return np.full(tuple(filter(lambda d:d!=0, seq.shape)), -np.inf)
        elif seq.shape[axis] == 1:
            return seq.take(0, axis)
        elif seq.shape[axis] == 2:
            return np.logaddexp2(seq.take(0, axis), seq.take(1, axis))
        else:
            # return np.logaddexp2(seq.take(0, axis), lse2(seq.take(np.arange(1, seq.shape[axis]), axis), axis))
            res = seq.take(0, axis)
            for k in np.arange(1, seq.shape[axis]):
                res = np.logaddexp2(res, seq.take(k, axis))
            return res
