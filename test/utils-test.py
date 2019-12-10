#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.utils import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import wave

def on_key(event:mpl.backend_bases.KeyEvent):
    if event.key == 'escape':
        plt.close()

#%% 分帧
print(split_frames(np.arange(30), 3, 2))

#%% 录音
data = recording(time=2)
# data = recording(2, 'w.pcm', 'w.wav')

plt.figure('Plot')
plt.connect('key_press_event', on_key)
plt.plot(data)
plt.show()



