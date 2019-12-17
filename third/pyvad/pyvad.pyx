#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pyvad

 - https://github.com/shiweixingcn/vad
"""

cimport pyvad
import numpy as np
cimport numpy as cnp

cdef class Vad:
    FrameLen = 256
    cdef pyvad.VadVars* _c_vad

    def __cinit__(self):
        pyvad.wb_vad_init(&self._c_vad)

    def __dealloc__(self):
        pyvad.wb_vad_exit(&self._c_vad)

    def reset(self):
        pyvad.wb_vad_reset(self._c_vad)

    def process(self, cnp.ndarray[cnp.int16_t, ndim=1] data):
        size = data.shape[0] // self.FrameLen
        res = np.zeros(size, dtype=np.int8)
        power_sum = np.zeros(size, dtype=np.float64)
        cdef int cres
        cdef double cpower_sum
        cdef int16_t* pdata = &data[0]
        for k in range(size):
            pyvad.wb_vad(self._c_vad, &pdata[k * self.FrameLen], &cres, &cpower_sum)
            res[k] = cres
            power_sum[k] = cpower_sum 
        return (res, power_sum)

    def set_pow_low(self, float power_sum):
        pyvad.wb_vad_set_pow_low(self._c_vad, power_sum)

    def set_pitch_tone_threshold(self, float pitch_tone_thr):
        pyvad.wb_vad_set_pow_pitch_tone_thr(self._c_vad, pitch_tone_thr)

    def set_pitch_tone_detection(self, float gain):
        pyvad.wb_vad_pitch_tone_detection(self._c_vad, gain)
