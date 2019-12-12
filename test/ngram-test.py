#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DataSet:

 - THCHS-30, http://www.openslr.org/18/
"""

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.ngram import *
import glob
import math

order = 3

# 获取数据索引
thchs30 = './data_thchs30/trn'
trn_files = glob.glob(thchs30 + '/*.wav.trn')

#%% 字符串序列化
tokens = []
for k in range(len(trn_files)):
# for k in range(1):
    with open(trn_files[k], mode='r', encoding='utf-8') as fp:
        line = fp.readlines()
        tokens.append(Tokenizer.get_tokens(line[0].strip()))
# print(tokens)

#%% NGramCounter统计
nc = NGramCounter(order, tokens)
# print(nc)
# print(nc.ngrams)

#%% NGram模型
ng = NGramModel(nc)
# print(ng._logprob("阳春", ("绿", "是")))
# print(ng._logprob("用时", ("绿", "是")))
token = Tokenizer.get_tokens("今天 如此 娃哈哈 过去 甚至 延至 三十一日 违反 县级 另外 原则上 排名 看不到 的 情况")
# token = tokens[1]
print(ng.calc_ppl(token))


#%% ARPA格式模型
NGramModelARPA().save(ng, 'trigram.lm')
new_ng = NGramModel(NGramModelARPA().load('trigram.lm'))
print(new_ng.calc_ppl(token))
