# -*- coding: utf-8 -*-
# range_calculator.py
import numpy as np
from signal_processor import freq_process
from config import FS, N_FFT, N_CZT, START_POINT, FRAME_LEN as SAMPLES_PER_FRAME, CAMERA_PARAMS

def process_all_frames(Data_1550, Data_1560, camera_no=2, progress_callback=None, log_callback=None):
    """
    主循环：逐帧计算频率和距离，完全复现 MATLAB 逻辑。

    参数：
        Data_1550, Data_1560: 二维数组，形状 (采样点数, 帧数)
        camera_no: 相机编号 (1 或 2)
        progress_callback: 可选，接收 (percent, msg) 的回调函数
        log_callback: 可选，接收字符串的回调函数
    返回：
        Dist, Avg_40, Meas1550, Meas1560, fpm_1550, fpm_1560, stats, ad_1550_last, ad_1560_last
    """
    params = CAMERA_PARAMS[camera_no]
    Lr_1550 = params['Lr_1550']
    Lr_1560 = params['Lr_1560']
    L0 = params['L0']

    n_measurements = Data_1550.shape[1]
    if log_callback is not None:
        log_callback(f"开始处理 {n_measurements} 次测量...")

    Meas1550 = np.zeros(n_measurements)
    Meas1560 = np.zeros(n_measurements)
    Dist = np.zeros(n_measurements)
    fpm_1550 = np.zeros(n_measurements)
    fpm_1560 = np.zeros(n_measurements)

    for k in range(n_measurements):
        samp1 = Data_1550[START_POINT: START_POINT + SAMPLES_PER_FRAME, k]
        samp2 = Data_1560[START_POINT: START_POINT + SAMPLES_PER_FRAME, k]

        # 1550 通道处理
        f_1550 = freq_process(FS, N_FFT, N_CZT, samp1, seri_numb=k + 1, is_drawing=False)
        Meas1550[k] = Lr_1550 / N_FFT * f_1550
        fpm_1550[k] = f_1550 * FS / N_FFT

        # 1560 通道处理
        f_1560 = freq_process(FS, N_FFT, N_CZT, samp2, seri_numb=k + 1, is_drawing=False)
        Meas1560[k] = Lr_1560 / N_FFT * f_1560
        fpm_1560[k] = f_1560 * FS / N_FFT

        # 距离计算（单位 mm）
        Dist[k] = ((Meas1550[k] + Meas1560[k]) / 2 - L0) / 1000

        # 进度回调（每帧）
        if progress_callback is not None:
            percent = int((k + 1) / n_measurements * 100)
            progress_callback(percent, f"已处理 {k + 1}/{n_measurements} 次测量")

        # 日志回调（每 40 帧）
        if log_callback is not None and (k + 1) % 40 == 0:
            log_callback(f"已解算 {k + 1}/{n_measurements} 帧")

    # 循环结束后，进度回调完成
    if progress_callback is not None:
        progress_callback(100, "解算完成")

    # 计算 40 组平均值（剔除最大最小值）
    avg_step = 40
    n_avg = n_measurements // avg_step
    Avg_40 = np.zeros(n_avg)
    cnt = 0
    for i in range(n_avg):
        start_idx = i * avg_step
        end_idx = start_idx + avg_step
        segment = Dist[start_idx:end_idx]
        if len(segment) == avg_step:
            sum_cut = np.sum(segment) - np.max(segment) - np.min(segment)
            Avg_40[cnt] = sum_cut / (avg_step - 2)
            cnt += 1
    Avg_40 = Avg_40[:cnt]

    # 统计信息
    stats = {
        'freq_mean_1550': np.mean(fpm_1550),
        'freq_mean_1560': np.mean(fpm_1560),
        'freq_std_1550': np.std(fpm_1550),
        'freq_std_1560': np.std(fpm_1560),
        'dist_mean': np.mean(Avg_40) if len(Avg_40) > 0 else np.nan
    }

    # 截取最后一次测量的 AD 波形（可用于后续分析）
    ad_1550_last = Data_1550[:, -1]
    ad_1560_last = Data_1560[:, -1]

    if log_callback is not None:
        log_callback("处理完成。")

    return Dist, Avg_40, Meas1550, Meas1560, fpm_1550, fpm_1560, stats, ad_1550_last, ad_1560_last