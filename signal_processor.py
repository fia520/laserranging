# -*- coding: utf-8 -*-
# signal_processor.py
import numpy as np
from scipy.signal import czt

def freq_process(fs, n_fft, n_czt, data, seri_numb=0, is_drawing=False, return_spectra=False):
    """
    对单帧数据进行 FFT 粗测 + CZT 精测，返回峰值频率位置（采样点序号）。

    新增参数:
        return_spectra: bool，若为 True，则返回 (freq_pos, spectra_dict)
                        spectra_dict 包含 'P1_fft', 'f1', 'P2', 'fz'
    """
    # ---------- 粗测：FFT ----------
    Yfft = np.fft.fft(data, n=n_fft)
    P2_fft = np.abs(Yfft) / n_fft
    P1_fft = P2_fft[:n_fft // 2 + 1]
    P1_fft[1:-1] = 2 * P1_fft[1:-1]
    f1 = fs * np.arange(n_fft // 2 + 1) / n_fft

    search_range = slice(199, n_fft // 2)
    pm_1 = np.argmax(P1_fft[search_range]) + 199
    fm_1 = fs * pm_1 / n_fft
    fm_0 = fs * (pm_1 - 20) / n_fft
    delta_fm = fs * 40 / n_fft

    # ---------- 精测：CZT ----------
    w = np.exp(-1j * 2 * np.pi * delta_fm / (n_czt * fs))
    a = np.exp(1j * 2 * np.pi * fm_0 / fs)
    z = czt(data[:n_czt], m=n_czt, w=w, a=a)
    P2 = np.abs(z)
    kn = np.arange(n_czt) / n_czt
    fz = delta_fm * kn + fm_0

    pm_czt = np.argmax(P2)
    freq_pos = pm_1 - 20 + pm_czt * 40 / n_czt

    if return_spectra:
        spectra = {
            'P1_fft': P1_fft,
            'f1': f1,
            'P2': P2,
            'fz': fz
        }
        return freq_pos, spectra
    else:
        return freq_pos
