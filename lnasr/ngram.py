#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
N元语言模型（N-Gram Model），使用ARPA格式保存模型。
"""

import sys,os
sys.path.append(os.getcwd() + '/../')

from lnasr.utils import Punctuation_Unicode_Set
from collections import defaultdict, Counter
import math
import re
from typing import List, Tuple

CharSpace = " "
CharLF = "\n"

class Tokenizer:
    """文本向量化"""
    Pnctt_Set = Punctuation_Unicode_Set

    @staticmethod
    def get_tokens(text:str, s:bool=True)->Tuple[str]:
        """将一段文本向量化

        :Parameters:
            - text: 文本，包含的中文标点当作空格处理
            - s: 文本向量化后的头尾是否添加<s>和</s>

        :Returns: 保存token的元组
        """
        line = ""
        for k in range(len(text)):
            if text[k] in Tokenizer.Pnctt_Set:
                line += " "
            else:
                line += text[k]
        if s:
            line = CharSpace.join(["<s>", line, "</s>"])
        tokens = tuple(line.split())
        return tokens

class NGramCounter(defaultdict):
    """NGram计数模型

    - `n`       即self.order，n元语法，一个ngram为(w1, w2, ..., w[n])
    - `self`    (w1 ~ w[n])的ngram计数之和，用dict保存，键为(w1 ~ w[n-1])，值为{w[n]: cnt}
    - `N`       (w1 ~ w[n-1], wi)的ngram计数之和
    - `V`       (w1 ~ w[n-1], wi)的ngram数量

    以3-gram为例：

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

    def __init__(self, order, tokens:Tuple[Tuple[str]]):
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
        return CharLF.join(strs)

    def __count(self, token:Tuple[str]):
        """统计向量化文本的NGram"""
        for k in range(self.order-1, len(token)):
            w = tuple(token[k-self.order+1:k])
            self[w][token[k]] += 1
            self.ngrams.add(w + (token[k],))

class NGramModel(dict):
    """中文NGram模型，这里使用3元模型，即TriGram"""

    DisCount = 0.7
    K = 0

    def __init__(self, ng):
        """初始化

        :Parameters:
            - ng: NGram计数模型或ARPA模型
        """
        self.order = ng.order
        if isinstance(ng, NGramCounter):
            self.prob = {}
            self.prob_bo = {}
            # 递归生成低阶NGram模型
            counter = ng
            self._estimate_prob(counter)
            if self.order > 1:
                self.backoff = NGramModel(counter.backoff)
                self._estimate_alpha()
            else:
                self.backoff = None
                self.prob_bo = {}
        elif isinstance(ng, NGramModelARPA):
            self.prob = ng.prob
            if self.order > 1:
                self.backoff = NGramModel(ng.backoff)
                # 从ARPA生成NGram模型时中，n-grams元的alpha系数，来自(n-1)-grams中
                self.prob_bo = ng.backoff.prob_bo
            else:
                self.backoff = None
                self.prob_bo = {}

    def _estimate_prob(self, counter:NGramCounter):
        """估计模型打折概率，基于（Katz Backoff）

        :Parameters:
            - ng: NGram计数模型
        """
        # TODO:使用Good-Turing，处理1-gram也没有w的情况，即计数为零的情况
        # 计算C(w1 ~ w[n]) > K的打折概率
        for context,wndict in counter.items():
            # 固定打折系数
            s = float(sum(wndict.values()))
            d = dict((wn, self.DisCount * float(cnt) / s) for wn,cnt in wndict.items() if cnt > self.K)
            if d:
                self[context] = d
        # 计算log概率
        self.prob = {}
        for context,wndict in self.items():
            for wn,p in wndict.items():
                self.prob[context + (wn,)] = math.log10(p)

    def _estimate_alpha(self):
        """估计模型打折系数，基于（Katz Backoff）"""
        # 计算C(w1 ~ w[n]) <= K的回退系数alpha(w1 ~ w[n-1])
        self.prob_bo = {}
        for context,wndict in self.items():
            beta = 1.0 - sum(self[context].values())
            alpha = 0.0
            for wn in self[context].keys():
                # 若C(w1 ~ w[n])>K，则必有C(w2 ~ w[n])>K
                alpha += self.backoff[context[1:]][wn]
            self.prob_bo[context] = math.log10(beta / (1.0 - alpha))

    def _estimate_smothing(self, counter:NGramCounter, smoothing):
        """估计模型概率

        :Parameters:
            - ng: NGram计数模型
        """
        if smoothing == None:
            # 无平滑（直接使用MLE）
            for context,wndict in counter.items():
                s = float(sum(wndict.values()))
                self[context] = dict((wn, float(cnt) / s) for wn,cnt in wndict.items())
        elif smoothing == 'AddOne':
            # 加1平滑
            for context,wndict in counter.items():
                s = float(sum(wndict.values()) + len(wndict))
                self[context] = dict((wn, float(cnt + 1.0) / s) for wn,cnt in wndict.items())

    def _logalpha(self, context:Tuple[str])->float:
        """计算回退权值alpha"""
        if context in self.prob_bo.keys():
            # 存在(w1 ~ w[n-1])
            return self.prob_bo[context]
        else:
            # 不存在(w1 ~ w[n-1])，直接回退到n-2
            return 0.0

    def _logprob(self, word:str, context:Tuple[str])->float:
        """使用回退法（Katz Backoff）估计一个ngram的log概率

        一个ngram为(context, word)，计算概率公式：log10(P(word | context))。

        :Parameters:
            - word: 第n个token，wn
            - context: 第1~n个token序列，(w1 ~ w[n-1])
        """
        w = context + (word,)
        if (w in self.prob.keys()) or self.order == 1:
            return self.prob[w]
        else:
            return self._logalpha(context) + self.backoff._logprob(word, context[1:])

    def calc_prob(self, sentence:Tuple[str])->float:
        """计算概率（probability）"""
        prob = 0.0
        for k in range(self.order-1, len(sentence)):
            prob += self._logprob(sentence[k], sentence[k-self.order+1:k])
        return prob

    def calc_ppl(self, sentence:Tuple[str])->float:
        """计算困惑度（perplexity）"""
        prob = self.calc_prob(sentence)
        ppl = math.pow(10, -prob/len(sentence))
        return ppl

class NGramModelARPA:
    """NGram的ARPA格式模型"""

    # LM标识字符串
    LmData       = "\\data\\"
    LmDataNum    = "ngram {}={}"
    LmDataNumPat = r'ngram \d*=\d*'
    LmNgrams     = "\\{}-grams:"
    LmNgramsPat  = r'\\(\d)*-grams:'
    LmEnd        = "\\end\\"
    # LM数据
    LmSections = {
        'comment': [],
        'data': [],
        'grams': []
        }

    def _init(self, model:NGramModel):
        """根据NGramModel初始化"""
        self.order = model.order
        self.prob = model.prob
        if self.order > 1:
            self.backoff = NGramModelARPA()
            self.backoff._init(model.backoff)
            # 在ARPA格式中，n-grams元的alpha系数，放在(n-1)-grams中
            self.backoff.prob_bo = model.prob_bo
        else:
            self.backoff = None

    def _lm_data(self, lm:dict):
        """生成ARPA模型数据"""
        if self.backoff != None:
            self.backoff._lm_data(lm)
        lm['data'].append(self.LmDataNum.format(self.order, len(self.prob)))
        grams = [self.LmNgrams.format(self.order)]
        try:
            for k,v in self.prob.items():
                line = "{}\t{}".format(v, CharSpace.join(k))
                try:
                    if k in self.prob_bo.keys():
                        line += "\t{}".format(self.prob_bo[k])
                except Exception as e:
                    pass
                grams.append(line)
        except Exception as e:
            pass
        lm['grams'].append(grams)

    def _lm_text(self):
        """生成ARPA模型文本"""
        # 生成LM数据
        lm = self.LmSections.copy()
        self._lm_data(lm)
        # 生成LM文本
        text = ""
        text += CharLF.join(lm['comment'])
        text += CharLF + self.LmData + CharLF
        text += CharLF.join(lm['data']) + CharLF
        for k in lm['grams']:
            text += CharLF + CharLF.join(k) + CharLF
        text += CharLF + self.LmEnd + CharLF
        return text

    def save(self, model:NGramModel, filename):
        """保存ARPA语言模型"""
        self._init(model)
        with open(filename, mode='w', encoding='utf-8') as fp:
            fp.write(self._lm_text())

    def _lm_parse(self, filename):
        """解析lm文件"""
        arpa = ""
        mods = []
        self.order = 0
        with open(filename, mode='r', encoding='utf-8') as fp:
            for line in fp:
                line = line.strip()
                # Empty
                if not line:
                    continue
                if arpa == "":
                    # Wait \data\
                    if line == self.LmData:
                        arpa = self.LmData
                elif arpa == self.LmData:
                    if re.match(self.LmDataNumPat, line) != None:
                        # get order
                        self.order += 1
                    elif re.match(self.LmNgramsPat, line) != None:
                        # init backoff
                        self.prob = {}
                        self.prob_bo = {}
                        mods.append(self)
                        for k in range(self.order - 1):
                            mods[k].backoff = NGramModelARPA()
                            mods[k].backoff.prob = {}
                            mods[k].backoff.prob_bo = {}
                            mods[k].backoff.order = mods[k].order - 1
                            mods.append(mods[k].backoff)
                        arpa = self.LmNgrams
                if arpa == self.LmNgrams:
                    ret = re.match(self.LmNgramsPat, line)
                    if ret != None:
                        # get n
                        nr = int(ret.group(1))
                    else:
                        if line == self.LmEnd:
                            # Wait \end\
                            arpa = self.LmEnd
                        else:
                            # get prob
                            seps = line.split()
                            p = float(seps[0])
                            w = tuple(seps[1:nr+1])
                            mods[self.order - nr].prob[w] = p
                            if len(seps) >= nr + 2:
                                pb = float(seps[nr+1])
                                mods[self.order - nr].prob_bo[w] = pb

    def load(self, filename):
        """加载ARPA语言模型"""
        self._lm_parse(filename)
        return self
