#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recognizer.
"""

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.utils import *
from lnasr.mfcc import MFCC
from lnasr.gmmhmm import GMMHMM
from lnasr.ngram import NGramCounter, NGramModel, NGramModelARPA
from lnasr.lexicon import Lexicon
import numpy as np
import h5py


class AcousticModel:
    """声学模型"""

    def __init__(self, modelfile):
        self.mfcc = MFCC()
        self.gmmhmm = GMMHMM(n=2, m=2, d=self.mfcc.D)
        self.gmmhmm.load(modelfile)

class LanguageModel:
    """语言模型"""

    def __init__(self, modelfile):
        self.ngram = NGramModel(NGramModelARPA().load(modelfile))

class Recognizer:
    """语音解码器"""

    def __init__(self, model:dict):
        """初始化

        :Parameters:
            - model: 包括声学模型、语言模型等参数
        """
        self.am = AcousticModel(model['am'])
        self.lm = LanguageModel(model['lm'])

    def recognize(self, data:np.ndarray)->str:
        """语音解码"""
        pass
