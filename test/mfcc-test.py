
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.mfcc import MFCC
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['font.family'] = 'Consolas'
mpl.rcParams['font.size'] = 11

def on_key(event:mpl.backend_bases.KeyEvent):
    if event.key == 'escape':
        plt.close()

def get_voice_raw(filepath)->np.ndarray:
    with open(filepath, 'rb') as fp:
        byte = fp.read()
        size = fp.tell() // 2
        vr = np.zeros((size), dtype=np.int16)
        for k in range(size):
            vr[k] = (byte[2*k]) | (byte[2*k+1]<<8)
    return vr

if __name__ == "__main__":
    mfcc = MFCC(
            fs=16000,
            frame_T=25e-3,
            frame_stride=10e-3,
            hpf_alpha=0.97,
            fft_N=512,
            mfbank_num=40,
            cepstrum_num=12
            )

    vr = get_voice_raw("data.raw").astype(dtype=np.float, copy=False)    # 语音信号数据
    frames_vr, frames_magnitude, frames_power, frames_cepstrum, features_mfcc = mfcc.calc_mfcc(vr)

    # 绘制Mel Filter Bank
    # mfbank = mfcc.create_filter_bank(mfcc.mfbank_num, mfcc.fft_N)
    # fig = plt.figure('Mel Filter Bank')
    # fig.canvas.mpl_connect('key_press_event', on_key)
    # ax = fig.add_subplot(1, 1, 1)
    # for k in range(mfcc.mfbank_num):
    #     ax.plot(mfbank[k, :], 'r')
    #     ax.plot(np.linspace(1, mfcc.fs//2, num=mfcc.fft_size), mfbank[k, :], 'r')
    # ax.plot(mfbank[30, :], 'r')
    # ax.plot(np.linspace(1, mfcc.fs//2, num=mfcc.fft_size), mfbank[1, :], 'r')
    # plt.show()

    fig = plt.figure('Signal')
    fig.canvas.mpl_connect('key_press_event', on_key)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.plot(vr)
    ax2.plot(frames_vr[100])
    ax3.plot(np.linspace(1, mfcc.fs//2, num=mfcc.fft_size), frames_magnitude[100])
    ax4.plot(np.linspace(1, mfcc.fs//2, num=mfcc.fft_size), frames_power[100])
    plt.show()

    fig = plt.figure('Features')
    fig.canvas.mpl_connect('key_press_event', on_key)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax1.imshow(frames_cepstrum, cmap=mpl.cm.jet)
    ax2.plot(frames_cepstrum[100], 'r-*')
    ax3.imshow(features_mfcc, cmap=mpl.cm.jet)
    plt.show()
