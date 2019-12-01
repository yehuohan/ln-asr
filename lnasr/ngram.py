#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
N元语言模型（N-Gram Model），使用ARPA格式保存模型。
"""

from .utils import Punctuation_Unicode_Set
from collections import defaultdict, Counter
import numpy as np
from typing import List, Tuple

class _Tokenizer:
    """文本向量化"""
    Pnctt_Set = Punctuation_Unicode_Set

    @staticmethod
    def get_tokens(text:str, s:bool=True)->np.ndarray:
        """将一段文本向量化

        :Parameters:
            - text: 文本，包含的中文标点当作空格处理
            - s: 文本向量化后的头尾是否添加<s>和</s>

        :Returns: 保存token的numpy数组
        """
        line = ""
        for k in range(len(text)):
            if text[k] in _Tokenizer.Pnctt_Set:
                line += " "
            else:
                line += text[k]
        if s:
            line = " ".join(["<s>", line, "</s>"])
        tokens = np.array(line.split(), dtype=np.chararray)
        return tokens

class NGramCounter(defaultdict):
    """NGram计数模型

    - `n`       即self.order，n元语法，一个ngram为(w1, w2, ..., w[n])
    - `self`    (w1 ~ w[n])的ngram计数之和，用dict保存，键为(w1 ~ w[n-1])，值为{w[n]: cnt}
    - `N`       (w1 ~ w[n-1], *)的ngram计数之和
    - `V`       (w1 ~ w[n-1], *)的ngram数量

    以3-gram为例，说明各变量作用：

    ::

        self = {
            (w1,w2) : {
                w3: 5,
                w4: 8,
                w5: 1
                ...
            },
            (w1,w3) : {
                w7: 2
                ...
            }
            ...
        }
        对于(w1, w2, *)有：
            num = 5 + 8 + 1 + 2
            N = 5 + 8 + 1
            V = 3

    """

    def __init__(self, order, tokens:List[np.ndarray]):
        """初始化

        默认生成低阶模型。

        :Parameters:
            - order: 模型阶数，比如1-grams,2-grams等
            - tokens: 训练文本
        """
        super().__init__(Counter)
        self.order = order
        self.ngrams = set()
        for txt in tokens:
            self.__count(txt)
        # 递归生成低阶计数模型
        if self.order > 1:
            self.backoff = NGramCounter(self.order - 1, tokens)
        else:
            self.backoff = None

    def __str__(self):
        """打印NGram计数结果"""
        strs = []
        strs.append("{}-grams".format(self.order))
        for k,v in self.items():
            strs.append(str(k) + " : " + str(v))
        if self.backoff != None:
            strs.append(str(self.backoff))
        return "\n".join(strs)

    def __count(self, token:np.ndarray):
        """统计向量化文本的NGram"""
        for k in np.arange(self.order-1, len(token)):
            w = tuple(token[k-self.order+1:k])
            self[w][token[k]] += 1
            self.ngrams.add(w + (token[k],))

class NGramModel(dict):
    """中文NGram模型，这里使用3元模型，即TriGram"""

    def __init__(self, order):
        """初始化"""
        self.order = order

    def estimate(self, smoothing):
        """估计模型概率"""
        if smoothing == None:
            # 无平滑（直接使用MLE）
            for context,wndict in self.counter.items():
                s = float(sum(wndict.values()))
                self[context] = dict((wn, float(cnt) / s) for wn,cnt in wndict.items())
        elif smoothing == 'AddOne':
            # 加1平滑
            for context,wndict in self.counter.items():
                s = float(sum(wndict.values()) + len(wndict))
                self[context] = dict((wn, float(cnt + 1.0) / s) for wn,cnt in wndict.items())
        # 生成ARPA中的prob和prob_bo
        self.prob = {}
        self.prob_bo = {}
        for context,wndict in self.items():
            for wn,p in wndict.items():
                self.prob[context + (wn,)] = np.log10(p)

    def _train(self, tokens:List[np.ndarray]):
        """训练NGram模型，并生成低阶模型"""
        self.counter = NGramCounter(self.order, tokens)
        self.estimate(None)
        # self.estimate('AddOne')
        # 递归生成低阶NGram模型
        if self.order > 1:
            self.backoff = NGramModel(self.counter.backoff.order)
            self.backoff._train(tokens)
        else:
            self.backoff = None

    def train(self, data:List[str]):
        """训练NGram模型"""
        tokens = []
        for txt in data:
            tokens.append(_Tokenizer.get_tokens(txt, True))
        self._train(tokens)

    def save_lm(self, filename):
        """生成ARPA语言模型"""
        saver = NGramModelSaver(self)
        saver.save(filename)

    def load_lm(self, filename):
        """加载ARPA语言模型"""
        saver = NGramModelSaver(self)
        saver.load(filename)

class NGramModelSaver:
    """NGram的lm模型生成"""

    # LM标识字符串
    LmData     = "\n\\data\\\n"
    LmDataNum  = "ngram {}={}"
    LmNgrams   = "\n\\{}-grams:"
    LmEnd      = "\n\\end\\\n"
    # LM数据
    LmSections = {
        'comment': [],
        'data': [],
        'grams': []
        }

    def __init__(self, model:NGramModel):
        """初始化"""
        self.model = model

    def _appen_model(self, m:NGramModel):
        """生成模型数据"""
        if m.backoff != None:
            self._appen_model(m.backoff)
        self.lm['data'].append(self.LmDataNum.format(m.order, len(m.counter.ngrams)))
        grams = [self.LmNgrams.format(m.order)]
        try:
            for k,v in m.prob.items():
                line = "{}\t{}".format(v, " ".join(k))
                try:
                    if k in m.prob_bo.keys():
                        line += "\t{}".format(m.prob_bo[k])
                except Exception:
                    pass
                grams.append(line)
        except Exception as e:
            pass
        self.lm['grams'].append(grams)

    def _make_model(self):
        """生成模型文本"""
        text = ""
        text += "\n".join(self.lm['comment'])
        text += self.LmData
        text += "\n".join(self.lm['data']) + "\n"
        for k in self.lm['grams']:
            text += "\n".join(k) + "\n"
        text += self.LmEnd
        return text

    def save(self, filename):
        """保存ARPA语言模型"""
        self.lm = self.LmSections.copy()
        self._appen_model(self.model)
        with open(filename, mode='w', encoding='utf-8') as fp:
            fp.write(self._make_model())

    def load(self, filename):
        """加载ARPA语言模型"""
        raise Exception("Not implement")
