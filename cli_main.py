#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激光测距解算工具 - 交互式命令行版本
用法：
  交互模式：直接运行 python cli_main.py
  命令行模式：python cli_main.py -i input.dat -o output.csv ...
"""
import argparse
import csv
import sys
import os
import cmd
import numpy as np
from data_reader import extract_ad_data
from range_calculator import process_all_frames
from config import FS, N_FFT, N_CZT, START_POINT, FRAME_LEN as SAMPLES_PER_FRAME, CAMERA_PARAMS

# ---------- 核心解算函数 ----------
def run_calculation(input_file, output_file, camera_no, mode='all', frame=0):
    """执行解算，single 模式直接输出，all 模式保存到文件"""
    print(f"读取数据: {input_file}")
    try:
        Data_1550, Data_1560, fCnt = extract_ad_data(input_file, camera_no)
    except Exception as e:
        print(f"数据读取失败: {e}")
        return False

    print(f"共 {fCnt} 帧数据")

    if mode == 'single':
        if frame < 1 or frame > fCnt:
            print(f"帧序号 {frame} 超出范围 (1~{fCnt})")
            return False
        k = frame - 1
        samp1 = Data_1550[START_POINT:START_POINT+SAMPLES_PER_FRAME, k]
        samp2 = Data_1560[START_POINT:START_POINT+SAMPLES_PER_FRAME, k]
        from signal_processor import freq_process
        freq_pos_1550, _ = freq_process(FS, N_FFT, N_CZT, samp1, return_spectra=True)
        freq_pos_1560, _ = freq_process(FS, N_FFT, N_CZT, samp2, return_spectra=True)
        params = CAMERA_PARAMS[camera_no]
        Meas1550 = params['Lr_1550'] / N_FFT * freq_pos_1550
        Meas1560 = params['Lr_1560'] / N_FFT * freq_pos_1560
        dist_raw = (Meas1550 + Meas1560) / 2 - params['L0']
        dist_mm = dist_raw / 1000.0
        freq_1550_hz = freq_pos_1550 * FS / N_FFT
        freq_1560_hz = freq_pos_1560 * FS / N_FFT

        # 直接输出结果
        print("\n========== 单次解算结果 ==========")
        print(f"  帧序号:   {frame}")
        print(f"  距离:     {dist_mm:.3f} mm")
        print(f"  1550频率: {freq_1550_hz:.2f} Hz")
        print(f"  1560频率: {freq_1560_hz:.2f} Hz")
        print("==================================\n")
        return True

    else:  # mode == 'all'
        (Dist, Avg_40, Meas1550, Meas1560, fpm_1550, fpm_1560,
         stats, ad_1550, ad_1560) = process_all_frames(
            Data_1550, Data_1560, camera_no=camera_no
        )
        frames = list(range(1, len(Dist)+1))
        dists = Dist.tolist()
        freqs_1550 = fpm_1550.tolist()
        freqs_1560 = fpm_1560.tolist()

        # 保存到文件
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'dist_mm', 'freq_1550_Hz', 'freq_1560_Hz'])
            for i in range(len(frames)):
                writer.writerow([frames[i], f"{dists[i]:.6f}", f"{freqs_1550[i]:.6f}", f"{freqs_1560[i]:.6f}"])
        print(f"结果已保存至 {output_file}")
        return True
# ---------- 交互式菜单（极简黑白版） ----------
class LaserMenu(cmd.Cmd):
    intro = (
        "\n===== 激光测距解算工具 v1.0 =====\n"
        "输入命令:\n"
        "  i <文件>   设置输入文件\n"
        "  o <文件>   设置输出文件\n"
        "  c <1|2>    设置相机号\n"
        "  m <all|single>  设置解算模式\n"
        "  f <数字>   设置帧序号（仅 single 模式）\n"
        "  s          显示当前设置\n"
        "  h          显示帮助（命令列表）\n"
        "  r          执行解算\n"
        "  q          退出\n"
    )
    prompt = ">> "

    def __init__(self):
        super().__init__()
        self.input_file = None
        self.output_file = "result.csv"
        self.camera_no = 2
        self.mode = "all"
        self.frame = 0
        self.show_settings()

    def show_settings(self):
        """显示当前设置（简短格式）"""
        print("\n--- 当前设置 ---")
        print(f"  输入文件: {self.input_file or '(未设置)'}")
        print(f"  输出文件: {self.output_file}")
        print(f"  相机号:   {self.camera_no}")
        print(f"  解算模式: {self.mode}", end="")
        if self.mode == 'single':
            print(f" (帧序号: {self.frame})", end="")
        print("\n")

    def do_i(self, arg):
        """设置输入文件: i <路径>"""
        if not arg.strip():
            arg = input("请输入输入文件路径: ").strip().strip('"').strip("'")
        else:
            arg = arg.strip().strip('"').strip("'")
        if not arg:
            print("未输入路径")
            return
        if not os.path.isfile(arg):
            print(f"文件不存在: {arg}")
            return
        self.input_file = arg
        print(f"输入文件已设置: {self.input_file}")
        self.show_settings()

    def do_o(self, arg):
        """设置输出文件: o <路径>"""
        if not arg.strip():
            arg = input("请输入输出文件路径: ").strip().strip('"').strip("'")
        else:
            arg = arg.strip().strip('"').strip("'")
        if not arg:
            print("未输入路径")
            return
        self.output_file = arg
        print(f"输出文件已设置: {self.output_file}")
        self.show_settings()

    def do_c(self, arg):
        """设置相机号: c 或 c <1|2>"""
        if arg.strip():
            # 直接传参模式
            if arg.strip() not in ('1','2'):
                print("相机号只能是 1 或 2")
                return
            self.camera_no = int(arg.strip())
            print(f"相机号已设置为: {self.camera_no}")
            self.show_settings()
            return

        # 交互选择模式
        print("\n请选择相机号:")
        print("  1. FM107 (相机1)")
        print("  2. FM108 (相机2)")
        choice = input("请输入数字 (1-2): ").strip()
        if choice not in ('1','2'):
            print("无效选择，请重新输入")
            return
        self.camera_no = int(choice)
        print(f"相机号已设置为: {self.camera_no}")
        self.show_settings()

    def do_m(self, arg):
        """设置解算模式: m 或 m <all|single>"""
        if arg.strip():
            # 直接传参模式
            arg = arg.strip().lower()
            if arg not in ('all','single'):
                print("模式只能是 all 或 single")
                return
            self.mode = arg
            print(f"解算模式已设置为: {self.mode}")
            # 如果设为 single 模式，立即提示输入帧序号
            if self.mode == 'single':
                self._prompt_frame()
            self.show_settings()
            return

        # 交互选择模式
        print("\n请选择解算模式:")
        print("  1. 全部解算 (all)")
        print("  2. 单次解算 (single)")
        choice = input("请输入数字 (1-2): ").strip()
        if choice == '1':
            self.mode = 'all'
        elif choice == '2':
            self.mode = 'single'
            # 立即提示输入帧序号
            self._prompt_frame()
        else:
            print("无效选择，请重新输入")
            return
        print(f"解算模式已设置为: {self.mode}")
        self.show_settings()

    def _prompt_frame(self):
        """提示用户输入帧序号"""
        while True:
            inp = input("请输入帧序号 (正整数): ").strip()
            try:
                val = int(inp)
                if val < 1:
                    raise ValueError
                self.frame = val
                print(f"帧序号已设置为: {self.frame}")
                break
            except:
                print("输入无效，帧序号必须为正整数，请重新输入")

    def do_f(self, arg):
        """设置帧序号: f <数字> (仅 single 模式有效)"""
        if not arg.strip():
            arg = input("请输入帧序号 (正整数): ").strip()
        else:
            arg = arg.strip()
        try:
            val = int(arg)
            if val < 1:
                raise ValueError
            self.frame = val
            print(f"帧序号已设置为: {self.frame}")
            self.show_settings()
        except:
            print("帧序号必须为正整数")

    def do_s(self, arg):
        """显示当前设置"""
        self.show_settings()

    def do_h(self, arg):
        """显示帮助（命令列表）"""
        print(self.intro)

    def do_r(self, arg):
        """执行解算"""
        if not self.input_file:
            print("错误：请先设置输入文件 (i)")
            return
        success = run_calculation(
            self.input_file,
            self.output_file,
            self.camera_no,
            self.mode,
            self.frame
        )
        if success:
            print("解算完成！")
        else:
            print("解算失败，请检查参数")
        self.show_settings()

    def do_q(self, arg):
        """退出程序"""
        print("再见！")
        return True

    def complete_i(self, text, line, begidx, endidx):
        return [f for f in os.listdir('.') if f.endswith('.dat') and f.startswith(text)]

# ---------- 命令行模式入口 ----------
def parse_args():
    parser = argparse.ArgumentParser(description="激光测距解算命令行工具")
    parser.add_argument("-i", "--input", help="输入 .dat 文件路径")
    parser.add_argument("-o", "--output", default="result.csv", help="输出 CSV 文件路径")
    parser.add_argument("-c", "--camera", type=int, default=2, choices=[1,2], help="相机编号")
    parser.add_argument("-m", "--mode", default="all", choices=["all","single"], help="解算模式")
    parser.add_argument("-f", "--frame", type=int, default=0, help="单次解算时的帧序号")
    return parser.parse_args()

def main():
    if '-i' not in sys.argv and '--input' not in sys.argv:
        try:
            LaserMenu().cmdloop()
        except KeyboardInterrupt:
            print("\n用户中断，退出。")
        return

    args = parse_args()
    if not args.input:
        print("错误：请指定输入文件（使用 -i 参数）")
        sys.exit(1)
    run_calculation(args.input, args.output, args.camera, args.mode, args.frame)

if __name__ == '__main__':
    main()