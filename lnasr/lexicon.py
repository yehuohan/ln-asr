#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
发音词典（Pronunciation Lexicon）
"""

import numpy as np
from typing import Tuple

class Lexicon(dict):
    """发音词典，连接声学模型和语言模型"""

    def __init__(self):
        pass

    def map(self, state:np.ndarray)->Tuple[str]:
        pass
