#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from vad import *
import matplotlib as mpl
import matplotlib.pyplot as plt

def on_key(event:mpl.backend_bases.KeyEvent):
    if event.key == 'escape':
        plt.close()

def test_VadLtsd(wd, fs):
    wd = np.array(wd / (65536.0/2), dtype=np.double)
    vad = VadLtsd(
            freq=fs,
            winsize=1024,
            stepsize=512,
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

def test_VadwebRTC(wd, fs):
    res = VadWebRTC(wd, mode=0)
    plt.figure('Plot')
    res_points = np.arange(len(res)) * fs * 10e-3
    plt.connect('key_press_event', on_key)
    plt.plot(wd/65536.0/2, linewidth=0.5)
    plt.plot(res_points, res, 'r')
    plt.show()


if __name__ == "__main__":
    wd, fs = read_wave("data.wav")
    test_VadLtsd(wd, fs)
    test_VadwebRTC(wd, fs)
