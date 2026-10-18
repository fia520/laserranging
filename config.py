# -*- coding: utf-8 -*-
# config.py
import numpy as np

# 相机参数（根据 cameraNo 切换）
CAMERA_PARAMS = {
    2: {  # 正样FM108 2026年7月29日 长距离标定
        'Lr_1550': 30223894.7378,
        'Lr_1560': 29936095.2866,
        'L0': 1519145
    },
    1: {  # FM107 26年6月27日标定
        'Lr_1550': 30224769.5303,
        'Lr_1560': 29937904.2684,
        'L0': 1600000
    }
}

# 默认相机号
DEFAULT_CAMERA_NO = 2

# 信号处理参数
FS = 1.080e6          # 采样率 (Hz)
N_FFT = 2**14         # FFT 点数
N_CZT = 2**13         # CZT 点数
START_POINT = 5000    # 数据起始采样点
FRAME_LEN = 8192      # 每次处理的采样点数（与MATLAB一致）

# 其他常数
COL = 200
LINES = 80