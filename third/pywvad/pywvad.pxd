#!/usr/bin/env python3

from libc.stdint cimport int16_t, int32_t

cdef extern from "vad/vad_core.h":
    ctypedef struct VadInstT:
        pass

    int WebRtcVad_InitCore(VadInstT* v)
    int WebRtcVad_set_mode_core(VadInstT* v, int16_t ohmax1, int16_t ohmax2, int16_t lthr, int16_t gthr)

    # Hz   : 0        , 1        , 2
    # 8KHz : 80/10ms  , 160/20ms , 240/30ms
    # 16KHz: 160/10ms , 320/20ms , 480/30ms
    # Only support 16Bits, 16KHz, 10ms -> 320bytes
    # 10ms*16KHz -> 160 samples(16bit)/frame -> 320 bytes/frame
    int WebRtcVad_CalcVad8khz(VadInstT* v, const int16_t* speech_frame, size_t frame_length);
    int WebRtcVad_CalcVad16khz(VadInstT* v, const int16_t* speech_frame, size_t frame_length)
    int WebRtcVad_CalcVad32khz(VadInstT* v, const int16_t* speech_frame, size_t frame_length);
    int WebRtcVad_CalcVad48khz(VadInstT* v, const int16_t* speech_frame, size_t frame_length);
