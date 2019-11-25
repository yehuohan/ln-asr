#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
就用HMM用于中文分词。

训练数据集可到（http://sighan.cs.uchicago.edu/bakeoff2005/）下载。
"""

import sys,os
sys.path.append(os.getcwd() + '/../../')

from lnasr.hmm import HMM
from lnasr.utils import Punctuation_Unicode_Set
import numpy as np
import h5py

class DataSet():
    """训练数据集预处理"""

    # 字符类型
    Type_Left        = 0
    Type_Space       = 1
    Type_Punctuation = 2
    Type_Character   = 3
    Type_Right       = 4
    # 特殊字符
    Pnctt_Set = Punctuation_Unicode_Set
    # 字符状态
    State_Begin  = 'B'      # 'B': Begin
    State_Member = 'M'      # 'M': Member
    State_End    = 'E'      # 'E': End
    State_Single = 'S'      # 'S': Single
    # Type_Character状态表（行为前字符，列后后字符）
    State_Table_Character = [
        [' ', 'S', 'S', 'B', 'S'],
        [' ', 'S', 'S', 'B', 'S'],
        [' ', 'S', 'S', 'B', 'S'],
        [' ', 'E', 'E', 'M', 'E'],
        ]

    def __init__(self, path, data_type='icwb2'):
        self.path = path
        self.type = data_type

    def type_ch(self, ch):
        """判断字符类型"""
        if ord(ch) == ord(' '):
            return self.Type_Space
        elif ch in self.Pnctt_Set:
            return self.Type_Punctuation
        else:
            return self.Type_Character

    def read_icwb2(self)->list:
        """读取icwb2 training数据"""
        training_data = None
        with open(self.path, 'r', encoding='utf-8') as fp:
            training_data = fp.readlines()
            training_data = list(map(lambda k: k.strip(), training_data))
        return training_data

    def mark_icwb2(self, data)->dict:
        """自动标注icwb2数据集状态

        Type_Space不需要标记，Type_Punctuation为State_Single；
        Type_Character需要根据前后字符类型来标记状态：

        ::

              S P C R
            L S S B S
            S S S B S
            P S S B S
            C E E M E

        :Parameters:
            - data: icwb2基于空格分割的数据集
        """
        T = len(data)
        assert(T > 1)
        txt = ''
        state = ''
        for k, ch in enumerate(data):
            cc = self.type_ch(ch)
            if cc == self.Type_Punctuation:
                txt += ch
                state += self.State_Single
            elif cc == self.Type_Character:
                txt += ch
                cl = self.Type_Left if k == 0 else self.type_ch(data[k-1])
                cr = self.Type_Right if k == T-1 else self.type_ch(data[k+1])
                state += self.State_Table_Character[cl][cr]
        return {'data': txt, 'state' : state}

    def get_marked_data(self)->dict:
        """获取标注训练数据"""
        training_data = self.read_icwb2()
        for data in training_data:
            if len(data) > 1:
                yield self.mark_icwb2(data)

class Seg(HMM):
    """基于HMM的分词类"""

    # 状态映射
    States = {
            DataSet.State_Begin  : 0,
            DataSet.State_Member : 1,
            DataSet.State_End    : 2,
            DataSet.State_Single : 3,
        }
    States_Inv = [DataSet.State_Begin, DataSet.State_Member, DataSet.State_End, DataSet.State_Single]

    def __init__(self, A:np.matrix=None, B:np.matrix=None, pi:np.ndarray=None):
        super().__init__(len(self.States), 65536, A, B, pi, np.double)

    def _map_data(self, data:str)->np.ndarray:
        """将观测字符映射成下标状态"""
        return np.array(list(map(lambda x: ord(x), data)))

    def _map_state(self, state:str)->np.ndarray:
        """将字符状态映射成下标状态"""
        return np.array(list(map(lambda x: self.States[x], state)))

    def _map_state_inv(self, state:np.ndarray)->str:
        """将字符状态下标反映射成状态字符"""
        txt = ''
        for k in state:
            txt += self.States_Inv[k]
        return txt

    def reset(self):
        """初始化训练参数"""
        self.A = np.zeros((self.n, self.n), dtype=np.double)
        self.B = np.zeros((self.n, self.m), dtype=np.double)
        self.pi = np.zeros(self.n, dtype=np.double)

    def save(self, filename):
        """保存模型参数"""
        f = h5py.File(filename, 'w')
        f.create_dataset('A', data=self.A)
        f.create_dataset('B', data=self.B)
        f.create_dataset('pi', data=self.pi)
        f.close()

    def load(self, filename):
        """加载模型参数"""
        f = h5py.File(filename, 'r')
        self.A = f['A']
        self.B = f['B']
        self.pi = f['pi']

    def train(self, datas):
        """中文分词参数训练

        通过频数统计训练参数
        """
        self.reset()
        for ds in datas:
            data = self._map_data(ds['data'])
            state = self._map_state(ds['state'])
            for k in range(len(data)-1):
                self.A[state[k]][state[k+1]] += 1
            for k in range(len(data)):
                self.B[state[k]][data[k]] += 1
            self.pi[state[0]] += 1
        self.A /= np.sum(self.A, axis=1).reshape(self.n, 1)
        self.B += 1
        self.B /= np.sum(self.B, axis=1).reshape(self.n, 1)
        self.pi /= np.sum(self.pi)

    def test(self, test_data):
        """测试中文分词"""
        # decode
        T = len(test_data)
        obs = self._map_data(test_data)
        path = self.decode(obs)
        return path

def print_data_state(data:str, state:str):
    """显示数据集和状态"""
    widths = [
        (126,    1), (159,    0), (687,     1), (710,   0), (711,   1),
        (727,    0), (733,    1), (879,     0), (1154,  1), (1161,  0),
        (4347,   1), (4447,   2), (7467,    1), (7521,  0), (8369,  1),
        (8426,   0), (9000,   1), (9002,    2), (11021, 1), (12350, 2),
        (12351,  1), (12438,  2), (12442,   0), (19893, 2), (19967, 1),
        (55203,  2), (63743,  1), (64106,   2), (65039, 1), (65059, 0),
        (65131,  2), (65279,  1), (65376,   2), (65500, 1), (65510, 2),
        (120831, 1), (262141, 2), (1114109, 1),
        ]

    def get_width(o):
        nonlocal widths
        if o == 0xe or o == 0xf:
            return 0
        for num, wid in widths:
            if o <= num:
                return wid
        return 1

    pdata = ''
    pstate = ''
    for k,s in enumerate(state):
        pdata += data[k]
        pstate += s
        if s == DataSet.State_End or s == DataSet.State_Single:
            pdata += ' '
            pstate += ' '
        if get_width(ord(data[k])) == 2:
            pstate += ' '
    print(pdata)
    print(pstate)


def test_seg_train():
    """使用Seg训练分词参数"""
    ds = DataSet("pku_training.utf8")
    training_data = ds.get_marked_data()
    # td1 = next(training_data)
    # print_data_state(td1['data'], td1['state'])
    hs = Seg()
    hs.train(training_data)
    hs.save("seg.hdf5")
    print(hs.A)
    print(hs.B)
    print(hs.pi)

def test_seg_test():
    """使用Seg进行分词"""
    test_data = "我不觉泪如泉涌，一下子我理解了许多，明白了许多。"
    true_data = "我 不觉 泪如泉涌 ， 一下子 我 理解 了 许多 ， 明白 了 许多 。"
    hs = Seg()
    hs.load("seg.hdf5")
    seg_data = hs.test(test_data)
    print(test_data)
    print(true_data)
    print_data_state(test_data, hs._map_state_inv(seg_data))

if __name__ == "__main__":
    # test_seg_train()
    test_seg_test()
