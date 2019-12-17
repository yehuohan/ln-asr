#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from libc.stdint cimport int16_t

cdef extern from "wb_vad.h":
    ctypedef struct VadVars:
        pass

    int wb_vad_init(VadVars ** vad)
    int wb_vad_reset(VadVars * vad)
    void wb_vad_exit(VadVars ** vad)
    void wb_vad(VadVars * st, const int16_t in_buf[], int* res, double* power_sum)
    void wb_vad_pitch_tone_detection(VadVars * vad, float gain)
    void wb_vad_set_pow_low(VadVars * vad, float power_sum)
    void wb_vad_set_pow_pitch_tone_thr(VadVars * vad, float pitch_tone_thr)
