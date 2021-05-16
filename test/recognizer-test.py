#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.recognizer import Recognizer

model = {
    'am' : 'gh.hdf5',
    'lm' : 'trigram.ln'
    }
r = Recognizer(model)
