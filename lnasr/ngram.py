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
