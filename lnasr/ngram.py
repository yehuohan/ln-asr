#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
N元语言模型（N-Gram Model），使用ARPA格式保存模型。
"""

from .utils import Punctuation_Unicode_Set
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
