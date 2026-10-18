# -*- coding: utf-8 -*-
# data_reader.py
import numpy as np

def extract_ad_data(filename: str, camera_no: int):
    """
    从 .dat 文件中提取1550nm和1560nm通道的AD数据。
    参数:
        filename: .dat 文件路径
        camera_no: 相机编号 (1 或 2)
    返回:
        Data_1550: (8192, framelen) uint16 数组
        Data_1560: (8192, framelen) uint16 数组
        framelen: 测量次数
    """
    raw = np.fromfile(filename, dtype=np.uint8).tobytes()
    file_size = len(raw)

    if camera_no == 1:
        frame_head = bytes([0xF0, 0xF6, 0x5A, 0xC9, 0x80])
    elif camera_no == 2:
        frame_head = bytes([0xF0, 0xF6, 0x5A, 0xC9, 0x84])
    else:
        raise ValueError("camera_no must be 1 or 2")

    # 查找帧头
    str_pos = []
    start = 0
    while True:
        pos = raw.find(frame_head, start)
        if pos == -1:
            break
        str_pos.append(pos)
        start = pos + 1
    if len(str_pos) == 0:
        raise ValueError("未找到帧头，请检查文件格式或相机号")

    # 过滤三角波下调数据
    valid_pos = [p for p in str_pos if p + 10 < file_size and (raw[p + 10] == 0 or raw[p + 10] == 170)]
    if len(valid_pos) == 0:
        raise ValueError("未找到有效的上调数据帧")

    # 重组数据到 bb0（每帧 430 字节）
    FRAME_BYTES = 430
    total_valid = len(valid_pos)
    bb0 = bytearray(total_valid * FRAME_BYTES)
    for i, p in enumerate(valid_pos):
        bb0[i*FRAME_BYTES : (i+1)*FRAME_BYTES] = raw[p : p+FRAME_BYTES]
    bb0 = bytes(bb0)

    # 在 bb0 中查找帧头
    str_pos2 = []
    start = 0
    while True:
        pos = bb0.find(frame_head, start)
        if pos == -1:
            break
        str_pos2.append(pos)
        start = pos + 1
    if len(str_pos2) == 0:
        raise ValueError("重新组织后未找到帧头")

    pos0 = str_pos2[0]
    packed = int.from_bytes(bb0[pos0+15 : pos0+17], 'big')
    if packed <= 0 or packed > 200:
        raise ValueError(f"packed 值异常 ({packed})")

    models = [bb0[p+8] for p in str_pos2]
    unique_models = sorted(set(models))
    serials = len(unique_models)
    if serials <= 0 or serials > 10:
        raise ValueError(f"数据类型数量异常 ({serials})")

    found_start = False
    for idx, p in enumerate(str_pos2):
        model = bb0[p+8]
        numb = int.from_bytes(bb0[p+17 : p+19], 'big')
        if model == unique_models[0] and numb == 1:
            start_idx = idx
            found_start = True
            break
    if not found_start:
        raise ValueError("未能找到完整数据包的起始帧")

    truncate_byte = str_pos2[start_idx]
    bb0 = bb0[truncate_byte:]

    str_pos3 = []
    start = 0
    while True:
        pos = bb0.find(frame_head, start)
        if pos == -1:
            break
        str_pos3.append(pos)
        start = pos + 1
    total3 = len(str_pos3)

    step = packed * serials
    framelen = total3 // step - 1
    if framelen <= 0 or framelen > 10000:
        raise ValueError(f"测量次数异常 ({framelen})")

    rows_per_frame = packed * 200
    Data_1550 = np.zeros((rows_per_frame, framelen), dtype=np.uint16)
    Data_1560 = np.zeros((rows_per_frame, framelen), dtype=np.uint16)

    for g in range(framelen):
        base_idx = g * step
        ch1550_count = 0
        ch1560_count = 0
        for offset in range(step):
            p = str_pos3[base_idx + offset]
            model = bb0[p + 8]
            if model == 34:
                target = Data_1550
                row = ch1550_count * 200
                ch1550_count += 1
            elif model == 66:
                target = Data_1560
                row = ch1560_count * 200
                ch1560_count += 1
            else:
                continue

            ad_start = p + 21
            if ad_start + 399 >= len(bb0):
                break
            for n in range(200):
                idx = ad_start + n * 2
                value = int.from_bytes(bb0[idx:idx+2], 'big')
                target[row + n, g] = value

    # Data_1550 = Data_1550[:8192, :]
    # Data_1560 = Data_1560[:8192, :]
    print(f"实际测量次数: {framelen}")

    return Data_1550, Data_1560, framelen