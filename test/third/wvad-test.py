#!/usr/bin/env python3

import sys,os
sys.path.append(os.getcwd() + '/../../')
sys.path.append(os.getcwd() + '/../../third/wvad')

from lnasr.utils import read_pcm
from wvad import Filter, Vad
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def on_key(event:mpl.backend_bases.KeyEvent):
    if event.key == 'escape':
        plt.close()

data = read_pcm('data-vad.raw')

# v = Vad(3, 6, 82, 285)
v = Vad(3, 6, 10, 30)
v.reset()
res = v.calc(data)

plt.figure()
plt.connect('key_press_event', on_key)
plt.plot(data/(2**15))
plt.plot(np.arange(len(res)) * 160, res, 'r')
plt.show()
