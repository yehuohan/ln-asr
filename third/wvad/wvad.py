#!/usr/bin/env python3

"""
使用python实现webrtc的vad算法。

 - 算法不进行技巧性优化，直接套公式计算，便于理解webrtc中vad算法；
 - 计算数据和相关参数转成float类型；
 - webrtc-vad源码：https://webrtc.googlesource.com/src/+/refs/heads/master/common_audio；
 - 参考：https://blog.csdn.net/Lucas23/article/details/84110584；
"""

import numpy as np

class Core:
    NumChannels = 6
    NumGaussians = 2
    MinEnergy = 10
    SpectrumWeight = np.array([6, 8, 10, 12, 14, 16], dtype=np.float)
    NoiseUpdateConst = 655.0 / (2**15)
    SpeechUpdateConst = 6554.0 / (2**15)
    BackEta = 154.0 / (2**8)
    MinimumDifference = np.array([544, 544, 576, 576, 576, 576],
            dtype=np.float) / (2**5)
    MinimumMean = np.array([640, 768], dtype=np.float) / (2**7)
    MaximumNoise = np.array([9216, 9088, 8960, 8832, 8704, 8576],
            dtype=np.float) / (2**7)
    MaximumSpeech = np.array([11392, 11392, 11520, 11520, 11520, 11520],
            dtype=np.float) / (2**7)
    NoiseDataWeights = np.array(
            [34, 62, 72, 66, 53, 25, 94, 66, 56, 62, 75, 103],
            dtype=np.float).reshape(NumGaussians, NumChannels) / (2**7)
    SpeechDataWeights = np.array(
            [48, 82, 45, 87, 50, 47, 80, 46, 83, 41, 78, 81],
            dtype=np.float).reshape(NumGaussians, NumChannels) / (2**7)
    NoiseDataMeans = np.array(
            [6738, 4892, 7065, 6715, 6771, 3369, 7646, 3863, 7820, 7266, 5020, 4362],
            dtype=np.float).reshape(NumGaussians, NumChannels) / (2**7)
    SpeechDataMeans = np.array(
            [8306, 10085, 10078, 11823, 11843, 6309, 9473, 9571, 10879, 7581, 8180, 7483],
            dtype=np.float).reshape(NumGaussians, NumChannels) / (2**7)
    NoiseDataStds = np.array(
            [378, 1064, 493, 582, 688, 593, 474, 697, 475, 688, 421, 455],
            dtype=np.float).reshape(NumGaussians, NumChannels) / (2**7)
    SpeechDataStds = np.array(
            [555, 505, 567, 524, 585, 1231, 509, 828, 492, 1540, 1079, 850],
            dtype=np.float).reshape(NumGaussians, NumChannels) / (2**7)
    MinStd = 384.0 / (2**7)
    MaxSpeechFrames = 6
    SmoothingDown = 6553.0 / (2**15)
    SmoothingUp = 32439.0  / (2**15)

    def __init__(self, kohm1:int, kohm2:int, klt:float, kgt:float):
        self.vad = 1    # speech active (=1)
        # gmm
        self.noise_means = np.zeros((Core.NumGaussians, Core.NumChannels), dtype=np.float)
        self.speech_means = np.zeros((Core.NumGaussians, Core.NumChannels), dtype=np.float)
        self.noise_stds = np.zeros((Core.NumGaussians, Core.NumChannels), dtype=np.float)
        self.speech_stds = np.zeros((Core.NumGaussians, Core.NumChannels), dtype=np.float)
        self.frame_counter = 0
        self.over_hang = 0
        self.num_of_speech = 0
        self.index_vector = np.zeros((Core.NumChannels, 16), dtype=np.float)
        self.low_value_vector = np.zeros((Core.NumChannels, 16), dtype=np.float)
        self.mean_value = np.zeros(Core.NumChannels, dtype=np.float)
        # filter
        self.downsampling_filter_states = np.zeros((2, 2), dtype=np.float)
        self.upper_state = np.zeros((5,2), dtype=np.float)
        self.lower_state = np.zeros((5,2), dtype=np.float)
        self.hp_filter_state = np.zeros(4, dtype=np.float)
        # parameters
        self.over_hang_max_1 = kohm1
        self.over_hang_max_2 = kohm2
        self.individual = klt
        self.total = kgt
        self.reset()

    def reset(self):
        self.vad = 1
        self.noise_means[:, :] = Core.NoiseDataMeans[:, :]
        self.speech_means[:, :] = Core.SpeechDataMeans[:, :]
        self.noise_stds[:, :] = Core.NoiseDataStds[:, :]
        self.speech_stds[:, :] = Core.SpeechDataStds[:, :]
        self.frame_counter = 0
        self.over_hang = 0
        self.num_of_speech = 0
        self.index_vector[:, :] = 0
        self.low_value_vector[:, :] = 10000.0 / (2**4)
        self.mean_value[:] = 1600.0 / (2**4)
        self.downsampling_filter_states[:, :] = 0
        self.upper_state[:, :] = 0
        self.lower_state[:, :] = 0
        self.hp_filter_state[:] = 0

class Filter:
    Upper = 0
    Lower = 1
    # 全通滤波器系数(upper, lower)
    AllPassCoefs = np.array([20972, 5571], dtype=np.float) / (2**15)
    HpZeroCoefs = np.array([6631, -13262, 6631], dtype=np.float) / (2**14)
    HpPoleCoefs = np.array([16384, -7756, 5620], dtype=np.float) / (2**14)
    OffsetVector = np.array([368, 368, 272, 176, 176, 176], dtype=np.float) / (2**4)    # 10*log10()

    @staticmethod
    def high_pass_filter(x:np.ndarray, s:np.ndarray)->np.ndarray:
        """二阶高通滤波器

        截止频率：80Hz
        b: HpZeroCoefs
        a: HpPoleCoefs
        s0 = x(n-1)
        s1 = x(n-2)
        s2 = y(n-1)
        s3 = y(n-2)
        y(n) = (b0 * x(n) + b1 * s0 + b2 * s1)
                         - (a1 * s2 + a2 * s3)
        H(z) = (b0 + b1 * z^(-1) + b2 * z^(-2)) / (a1 * z^(-1) + a2 * z^(-2))

        :Parameters:
            - x 音频信号
            - s 上一次滤波计算状态
        :Returns:
            - y 滤波后的信号
        """
        y = np.zeros_like(x, dtype=np.float)
        b = Filter.HpZeroCoefs
        a = Filter.HpPoleCoefs
        for n in range(len(x)):
            y[n] = (b[0] * x[n] + b[1] * s[0] + b[2] * s[1]) - \
                   (a[1] * s[2] + a[2] * s[3])
            s[1] = s[0]
            s[0] = x[n]
            s[3] = s[2]
            s[2] = y[n]
        return y

    @staticmethod
    def all_pass_filter(x:np.ndarray, c:float, s:np.ndarray)->np.ndarray:
        """二阶全通滤波器

        s0 = x(n-2) - c * y(n-2)
        s1 = x(n-1) - c * y(n-1)
        y(n) = x(n-2) - c * y(n-2) + c * x(n) = s0 + c * x(n)
        H(z) = (c + z^(-2)) / (1 + c * z^(-2))

        :Parameters:
            - x 音频信号
            - c 滤波系数
            - s 上一次滤波计算状态
        :Returns:
            - y 滤波后的信号
        """
        y = np.zeros_like(x, dtype=np.float)
        for n in range(len(y)):
            y[n] = s[0] + c * x[n]
            s[0] = s[1]
            s[1] = x[n] - c * y[n]
        return y

    @staticmethod
    def split_filter(x:np.ndarray, s0:np.ndarray, s1:np.ndarray):
        """split filter

        使用apf进行高通和低通滤波：
        A0(z) = all_pass_filter(c[0])
        A1(z) = all_pass_filter(c[1])
        Hpf(z) = (A1(z) - A0(z) * z^(-1)) / 2
        Lpf(z) = (A1(z) + A0(z) * z^(-1)) / 2
        hpf(n) = (a1(n) - a0(n-1)) / 2
        lpf(n) = (a1(n) + a0(n-1)) / 2

        将all_pass_filter序列计算公式代入（以lpf为例）：
        lph(n) = 1/2 * {
                 x(n-2) - c[1] * y(n-2) + c[1] * x(n)       -> 与a1偶数项有关
               + x(n-3) - c[0] * y(n-3) + c[0] * y(n-1)     -> 与a0奇数项有关
            }
        webrtc中的优化：针对c[1]，只计算偶数项
                        针对c[0]，只计算奇数项
                        计算全通滤波时，同时进行2倍下采样

        :Parameters:
            - x 音频信号
            - s0 上一次upper滤波计算状态
            - s1 上一次lower滤波计算状态
        :Returns:
            - (yh,yl) 高通和低通滤波后的信号
        """
        a0 = Filter.all_pass_filter(x, Filter.AllPassCoefs[0], s0)
        a1 = Filter.all_pass_filter(x, Filter.AllPassCoefs[1], s1)
        yh = (a1[1::2] - a0[0::2]) / 2
        yl = (a1[1::2] + a0[0::2]) / 2
        return (yh, yl)

    @staticmethod
    def log_energy(x:np.ndarray, offset:float, te:float):
        """计算频率子带能量

        webrtc中的offset和MinEnergy均用Q4类型，这里使用float类型。

        :Parameters:
            - x 子带滤波信号
            - te 总能量，10*log10()
        :Returns:
            - (te, log_energy) 子带能量，子带能量log_energy以10*log10()形式返回
        """
        log_energy = 0.
        energy = np.sum(x ** 2)
        if energy >= 0:
            log_energy = 10 * np.log10(energy)
        else:
            log_energy = offset
            return (te, log_energy)
        log_energy += offset

        # 总能量大于MinEnergy时，Gmm会计算概率，无需要再累加总能量
        if te <= Core.MinEnergy:
            tot_rshifts = np.log2(energy) - 14
            if tot_rshifts >= 0:
                te += Core.MinEnergy + 1.
            else:
                te += energy

        return (te, log_energy)

    @staticmethod
    def down_sampling(data:np.ndarray, core:Core)->np.ndarray:
        """使用全通滤波下采样至8KHz

        本质上是使用低通滤波(用全通滤波计算)。
        这里使用和webrtc中相同的优化流程。

        :Parameters:
            - data 16KHz, 10/20/30ms
        :Returns:
            - y 8KHz, 10/20/30ms
        """
        # 可以使用split_filter计算
        # return np.array(
        #         Filter.split_filter(
        #             data,
        #             core.downsampling_filter_states[0],
        #             core.downsampling_filter_states[1])[1],
        #         dtype=np.int16)
        x = data
        y = np.zeros(len(x)//2, dtype=np.float)
        c0 = Filter.AllPassCoefs[0]
        c1 = Filter.AllPassCoefs[1]
        s0 = core.downsampling_filter_states[0, 0]
        s1 = core.downsampling_filter_states[1, 0]
        for n in range(len(y)):
            # upper
            a0 = (s0 + c0 * x[2*n]) / 2
            s0 = x[2*n] - (c0 * a0) * 2
            # lower
            a1 = (s1 + c1 * x[2*n+1]) / 2
            s1 = x[2*n+1] - (c1 * a1) * 2
            y[n] = a0 + a1
        core.downsampling_filter_states[0, 0] = s0
        core.downsampling_filter_states[1, 0] = s1
        return y.astype(np.int16)

    @staticmethod
    def calc_features(data:np.ndarray, core:Core)->int:
        """计算子带特征

        频率子带划分：
            0.08K ~ 0.25K
            0.25K ~ 0.5K
            0.5K  ~ 1K
            1K    ~ 2K
            2K    ~ 3K
            3K    ~ 4K

        :Parameters:
            - data 8KHz, 10/20/30ms
        :Returns:
            - (total_energy, features) 信号总能量和子带能量特征，10*log10()
        """
        total_energy = 0.
        features = np.zeros(Core.NumChannels, dtype=np.float)
        hp120 = None
        lp120 = None
        hp60 = None
        lp60 = None

        # split at 2K : (2K~4K, 0K~2K)
        fb = 0      # 频率子带下标
        x = data    # 0K ~ 4K
        (hp120, lp120) = Filter.split_filter(x, core.upper_state[fb], core.lower_state[fb])

        # split at 3K : (3K~4K, 2K~3K)
        fb = 1
        x = hp120   # 2K ~ 4K
        (hp60, lp60) = Filter.split_filter(x, core.upper_state[fb], core.lower_state[fb])

        # energy in 3K~4K
        (total_energy, features[5]) = Filter.log_energy(hp60, Filter.OffsetVector[5], total_energy)
        # energy in 2K~3K
        (total_energy, features[4]) = Filter.log_energy(lp60, Filter.OffsetVector[4], total_energy)

        # split at 1K : (1K~2K, 0K~1K)
        fb = 2
        x = lp120   # 0K~2K
        (hp60, lp60) = Filter.split_filter(x, core.upper_state[fb], core.lower_state[fb])

        # energy in 1K~2K
        (total_energy, features[3]) = Filter.log_energy(hp60, Filter.OffsetVector[3], total_energy)

        # split at 0.5K : (0.5K~1K, 0K~0.5K)
        fb = 3
        x = lp60    # 0K~1K
        (hp120, lp120) = Filter.split_filter(x, core.upper_state[fb], core.lower_state[fb])

        # energy in 0.5K~1K
        (total_energy, features[2]) = Filter.log_energy(hp120, Filter.OffsetVector[2], total_energy)

        # split at 0.25K : (0.25K~0.5K, 0K~0.25K)
        fb = 4
        x = lp120   # 0K~0.5K
        (hp60, lp60) = Filter.split_filter(x, core.upper_state[fb], core.lower_state[fb])

        # energy in 0.25K~0.5K
        (total_energy, features[1]) = Filter.log_energy(hp60, Filter.OffsetVector[1], total_energy)

        # remove 0K~0.08K
        hp120 = Filter.high_pass_filter(lp60, core.hp_filter_state)

        # energy in 0.08K~0.25K
        (total_energy, features[0]) = Filter.log_energy(hp120, Filter.OffsetVector[0], total_energy)

        return (total_energy, features)

class Gmm:
    CompVar = 22005.0 / (2**10)

    @staticmethod
    def find_minimum(core:Core, feature_value, ch):
        """计算“最小值均值”

        low_index_vector: 保存每个子带特征最近100中最小的16个
        index_vector: 保存low_index_vector中每个子带特征的“年龄”
        mean_value:

        :Parameters:
            - feature_value 子带特征
            - ch 进行计算的子带下标
        :Returns:
            - 返回median
        """
        # 更新“年龄” index_vector
        for k in range(16):
            if core.index_vector[ch, k] != 100:
                core.index_vector[ch, k] += 1
            else:
                core.low_value_vector[ch, k:-1] = core.low_value_vector[ch, k+1:]
                core.index_vector[ch, k:-1] = core.index_vector[ch, k+1:]
                core.index_vector[ch, 15] = 101
                core.low_value_vector[ch, 15] = 10000.0 / (2**4)
        # 更新最小子带特征 low_value_vector（webrtc中使用二分法）
        for k in range(16):
            if feature_value < core.low_value_vector[ch, k]:
                core.low_value_vector[ch, k+1:] = core.low_value_vector[ch, k:-1]
                core.index_vector[ch, k+1:] = core.index_vector[ch, k:-1]
                core.low_value_vector[ch, k] = feature_value
                core.index_vector[ch, k] = 1
                break
        # 计算median
        current_median = 1600.0 / (2**4)
        alpha = 0.0
        if core.frame_counter > 2:
            current_median = core.low_value_vector[ch, 2]
        elif core.frame_counter > 0:
            current_median = core.low_value_vector[ch, 0]
        if core.frame_counter > 0:
            if current_median < core.mean_value[ch]:
                alpha = Core.SmoothingDown
            else:
                alpha = Core.SmoothingUp
        core.mean_value[ch] = \
                (alpha + 1.0 / (2**15)) * core.mean_value[ch] + \
                (1.0 - alpha) * current_median + \
                16384.0 / (2**(15+4))

        return core.mean_value[ch]

    @staticmethod
    def weighted_average(data:np.ndarray, offset:float, weights:np.ndarray):
        """计算加权均值

        data和weights应该为长度为2的数组，对应noise和speech
        """
        data += offset
        return np.sum(data * weights)

    @staticmethod
    def gaussian_probability(x:float, mean:float, std:float):
        """计算高斯分布概率

        webrtc利用定点算浮点，计算概率时，用了很多数值计算方法。

        :Parameters:
            - x 输入特征
            - mean 均值
            - std 标准差
        :Returns:
            - prob 返回概率
        """
        exp_value = 0.0
        tmp = (x - mean) * (x - mean) / (2.0 * std * std)
        if tmp < Gmm.CompVar:
            exp_value = np.exp(-tmp)

        return (1.0 / std) * exp_value

    @staticmethod
    def calc(core:Core, features:np.ndarray, total_power:float):
        """静音检测计算

        使用GMM计算一个feature为noise和speech的概率：
        m : 均值
        s : 标准差
        x : 一个feature的能量，即子带能量
        g(x, m, s, w) = sum(w * {1/s * exp(-(x-m)^2 / (2*s^2))})

        feature为noise和speech的概率：
        h0 = p(n|x) = g(x, m0, s0, w0)
        h1 = p(s|x) = g(x, m1, s1, w1)

        似然比检验(LRT)，检验speech模型和noise模型哪个更适合feature：
        lrt(x) = log2(h1/h0)

        基于极大似然估计更新GMM模型：
        _m = m + w * (x-m)/s^2 * c
        _s = s + w * 1/s * ((x-m)^2/s^2 - 1) * c

        :Parameters:
            - features 信号子带能量特征
            - total_power 总能量
        :Returns:
            静音检测结果
        """
        # variables
        vadflag = 0
        ngprvec = np.zeros((Core.NumGaussians, Core.NumChannels), dtype=np.float)
        sgprvec = np.zeros((Core.NumGaussians, Core.NumChannels), dtype=np.float)
        noise_prob = np.zeros(Core.NumGaussians, np.float)
        speech_prob = np.zeros(Core.NumGaussians, np.float)
        sum_log_likelihood_ratios = 0.0
        # parameters
        overhead1 = core.over_hang_max_1
        overhead2 = core.over_hang_max_2
        individualTest = core.individual
        totalTest = core.total

        if total_power > Core.MinEnergy:
            """
            计算概率子带特征为noise和speech的混合高斯概率
            """
            for ch in range(Core.NumChannels):
                for k in range(Core.NumGaussians):
                    noise_prob[k] = Core.NoiseDataWeights[k, ch] * \
                            Gmm.gaussian_probability(
                                    features[ch],
                                    core.noise_means[k, ch],
                                    core.noise_stds[k, ch])
                    speech_prob[k] = Core.SpeechDataWeights[k, ch] * \
                            Gmm.gaussian_probability(
                                    features[ch],
                                    core.speech_means[k, ch],
                                    core.speech_stds[k, ch])
                h0 = np.sum(noise_prob)
                h1 = np.sum(speech_prob)
                # 计算子带特征的对数似然比: log2(Pr{X|H1} / Pr{X|H0}).
                shifts_h0 = 31 if h0 <= 0.0 else (31 - 27 - np.log2(h0))
                shifts_h1 = 31 if h1 <= 0.0 else (31 - 27 - np.log2(h1))
                log_likelihood_ratio = shifts_h0 - shifts_h1
                # log_likelihood_ratio = np.log2(h1) - np.log2(h0)
                # 计算加权对数似然比
                sum_log_likelihood_ratios += log_likelihood_ratio * Core.SpectrumWeight[ch]
                # 任何一个子带的似然比大于阈值，认为是speech
                if log_likelihood_ratio * 4 > individualTest:
                    vadflag = 1
                if h0 > 0:
                    ngprvec[0, ch] = noise_prob[0] / h0
                    ngprvec[1, ch] = 1.0 - ngprvec[0, ch]
                else:
                    ngprvec[0, ch] = 1.0
                if h1 > 0:
                    sgprvec[0, ch] = speech_prob[0] / h1
                    sgprvec[1, ch] = 1.0 - sgprvec[0, ch]
            # 整帧的似然比大于阈值时，认为是speech
            if sum_log_likelihood_ratios >= totalTest:
                vadflag = 1

            Gmm.res = sum_log_likelihood_ratios

            """
            更新模型参数
            """
            for ch in range(Core.NumChannels):
                feature_minimum = Gmm.find_minimum(core, features[ch], ch)
                noise_global_mean = Gmm.weighted_average(
                        core.noise_means[:, ch], 0,
                        Core.NoiseDataWeights[:, ch])

                for k in range(Core.NumGaussians):
                    nmk = core.noise_means[k, ch]
                    nsk = core.noise_stds[k, ch]
                    smk = core.speech_means[k, ch]
                    ssk = core.speech_stds[k, ch]
                    deltaN = (features[ch] - nmk) / (nsk*nsk)
                    deltaS = (features[ch] - smk) / (ssk*ssk)

                    # 更新noise均值
                    tmp = nmk + \
                            (Core.NoiseUpdateConst * ngprvec[k, ch] * deltaN if vadflag == 0 else 0) + \
                            Core.BackEta * (feature_minimum - noise_global_mean)
                    core.noise_means[k, ch] = max(k + 5, min(tmp, 72 + k - ch))

                    if vadflag > 0:
                        # 更新speech均值
                        tmp = smk + Core.SpeechUpdateConst * sgprvec[k, ch] * deltaS
                        core.speech_means[k, ch] = max(Core.MinimumMean[k], min(tmp, (12800.0 + 640.0) / (2**7)))

                        # 更新speech标准差
                        ssk += sgprvec[k, ch] * (deltaS * (features[ch] - smk) - 1) * 0.1 / ssk
                        if ssk < Core.MinStd:
                            ssk = Core.MinStd
                        core.speech_stds[k, ch] = ssk
                    else:
                        # 更新noise标准差
                        nsk += ngprvec[k, ch] * (deltaN * (features[ch] - nmk) - 1) / nsk
                        if nsk < Core.MinStd:
                            nsk = Core.MinStd
                        core.noise_stds[k, ch] = nsk

                # Separate models if they are too close.
                noise_global_mean = Gmm.weighted_average(
                        core.noise_means[:, ch], 0,
                        Core.NoiseDataWeights[:, ch])
                speech_global_mean = Gmm.weighted_average(
                        core.speech_means[:, ch], 0,
                        Core.SpeechDataWeights[:, ch])
                diff = speech_global_mean - noise_global_mean
                if diff < Core.MinimumDifference[ch]:
                    t = Core.MinimumDifference[ch] - diff
                    speech_global_mean = Gmm.weighted_average(
                            core.speech_means[:, ch],
                            0.8 * t,
                            Core.SpeechDataWeights[:, ch])
                    noise_global_mean = Gmm.weighted_average(
                            core.noise_means[:, ch],
                            - 0.2 * t,
                            Core.NoiseDataWeights[:, ch])

                # Control that the speech & noise means do not drift to much.
                if speech_global_mean > Core.MaximumSpeech[ch]:
                    speech_global_mean -= Core.MaximumSpeech[ch]
                    core.speech_means[:, ch] -= speech_global_mean
                if noise_global_mean > Core.MaximumNoise[ch]:
                    noise_global_mean -= Core.MaximumNoise[ch]
                    core.noise_means[:, ch] -= noise_global_mean

            # 统计帧数
            core.frame_counter += 1

        # Smooth with respect to transition hysteresis.
        if 0 == vadflag:
            if core.over_hang > 0:
                vadflag = 2 + core.over_hang
                core.over_hang -= 1
            core.num_of_speech = 0
        else:
            core.num_of_speech += 1
            if core.num_of_speech > Core.MaxSpeechFrames:
                core.num_of_speech = Core.MaxSpeechFrames
                core.over_hang = overhead2
            else:
                core.over_hang = overhead1

        return vadflag

class Vad(Core):
    def __init__(self, kohm1:int, kohm2:int, klt:int, kgt:int):
        super(Vad, self).__init__(kohm1, kohm2, float(klt), float(kgt))

    def calc_vad(self, data:np.ndarray)->int:
        """静音检测

        :Parameters:
            - data 一帧音频信号，(16KHz, 16bit, 10ms)
        """
        frame = Filter.down_sampling(data, self)
        (total_power, feature_vector) = Filter.calc_features(frame, self)
        # print(total_power * (2**4), feature_vector * (2**4))
        self.vad = Gmm.calc(self, feature_vector, total_power)
        return self.vad

    def calc(self, data:np.ndarray)->np.ndarray:
        """静音检测

        :Parameters:
            - data 音频信号，只支持(16KHz, 16bit)
        """
        res_size = len(data) // 160
        res = np.zeros(res_size, dtype=np.int)
        for k in range(res_size):
            res[k] = self.calc_vad(data[k*160:(k+1)*160])
        return res
