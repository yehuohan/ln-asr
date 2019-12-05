#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DataSet:

 - THCHS-30, http://www.openslr.org/18/
"""

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.ngram import _Tokenizer
from lnasr.ngram import *
import numpy as np
import glob

order = 3

# 获取数据索引
thchs30 = './data_thchs30/data'
wav_files = glob.glob(thchs30 + '/*.wav')
trn_files = glob.glob(thchs30 + '/*.wav.trn')
txt_files = glob.glob(thchs30 + '/*.txt')

#%% 字符串序列化
tokens = []
data = []
for k in range(len(trn_files)):
# for k in range(1):
    with open(trn_files[k], mode='r', encoding='utf-8') as fp:
        line = fp.readlines()
        data.append(line[0].strip())
        tokens.append(_Tokenizer.get_tokens(line[0].strip()))
# print(tokens)

#%% NGramCounter统计
nc = NGramCounter(order, tokens)
# print(nc)
# print(nc.ngrams)

#%% NGram模型
ng = NGramModel(nc)
# ng.save_lm('trigram.lm')
# ng.load_lm('trigram.lm')
print(ng.calc_prob("阳春", ("绿", "是")))
print(ng.calc_prob("用时", ("绿", "是")))
