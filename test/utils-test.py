#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.utils import *
import numpy as np
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
data = recording(2, 'w.pcm', 'w.wav')
plt.figure('Plot')
plt.connect('key_press_event', on_key)
plt.plot(data)
plt.show()

#%% lse
import sys,os
sys.path.append(os.getcwd() + '/../')
from lnasr.utils import lse, lse2
import numpy as np

a = np.array([[[2, 2, 3],
               [2, 2, 3],
               [3, 3, 4]],
              [[2, 2, 3],
               [2, 2, 3],
               [3, 3, 4]]])
print(a.shape)
print(lse2(a))
print(lse2(a, axis=0))
print(lse2(a, axis=2))
print(lse2(lse2(a, axis=2), axis=0))
print(lse2(a, axis=(0, 2)))
i = np.array([0, 0, 1])
print(lse2(a[:, 2==i, :], axis=1))
print(lse2(a[:, 0==i, :], axis=1))
