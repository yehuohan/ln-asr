#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.getcwd() + '/../../')
sys.path.append(os.getcwd() + '/../../third/pyvad')

from lnasr.utils import read_pcm
from pyvad import Vad
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def on_key(event:mpl.backend_bases.KeyEvent):
    if event.key == 'escape':
        plt.close()

pdata = read_pcm('data-vad.raw')

vad = Vad()
vad.set_pow_low(100000000.0)
# vad.set_pow_low(10000000000.0)
plen = vad.FrameLen
pres, psum = vad.process(pdata)

plt.figure('Plot')
plt.connect('key_press_event', on_key)
plt.plot(pdata / (65536.0/2))
plt.plot(np.arange(len(pres)) * plen, pres, 'r')
plt.plot(np.arange(len(pres)) * plen, psum / psum.max(), 'y')
plt.show()
