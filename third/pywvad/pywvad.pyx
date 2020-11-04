#!/usr/bin/env python3

cimport pywvad
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int16_t, int32_t

cdef class Vad:
    cdef pywvad.VadInstT _c_vad

    def reset(self, int16_t k1, int16_t k2, int16_t kl, int16_t kg):
        """重置vad

        每次开始新一次会话的vad，需要重新reset。

        :Parameters:
            - k1 平滑参数（语音连续说话一段时间，则认为之后一定帧数也为说话）
            - k2 平滑参数（语音连续说话一段时间，则认为之后一定帧数也为说话）
            - kl 子带似然比阈值
            - kg 累加似然比阈值
        """
        pywvad.WebRtcVad_InitCore(&self._c_vad)
        pywvad.WebRtcVad_set_mode_core(&self._c_vad, k1, k2, kl, kg)

    def process(self, cnp.ndarray[cnp.int16_t, ndim=1] data):
        """vad

        :Parameters:
            - buf 音频数据（16KHz, 16Bits）
            - buf_len 应当为160（实际上会直接使用160，而不会使用此参数）

        :Returns:
            - 0 表示静音
            - >0 表示语音
        """
        cdef const int16_t* pdata = &data[0]
        n = data.shape[0] // 160
        res = np.zeros(n, dtype=np.int)
        for k in range(n):
            res[k] = pywvad.WebRtcVad_CalcVad16khz(&self._c_vad, &pdata[k * 160], 160)
        return res
