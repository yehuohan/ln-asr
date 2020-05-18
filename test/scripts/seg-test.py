#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.getcwd() + '/../../')
sys.path.append(os.getcwd() + '/../../scripts/seg')

from seg import *

def test_seg_train():
    """使用Seg训练分词参数"""
    ds = DataSet("pku_training.utf8")
    training_data = ds.get_marked_data()
    td1 = next(training_data)
    # pds = print_data_state(td1['data'], td1['state'])
    # print(pds[0])
    # print(pds[1])
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
    pds = print_data_state(test_data, hs._map_state_inv(seg_data))
    print(pds[0])
    print(pds[1])

if __name__ == "__main__":
    # test_seg_train()
    test_seg_test()
