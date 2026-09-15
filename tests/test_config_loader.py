#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
超随意配置热加载测试
特点：YAML 里写中文键名、逗号分隔列表、无缩进
"""

import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中 —— 必须在 import config 之前执行
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
from config.config_manager import config


def main():
    print("=" * 60)
    print("热插拔配置中心测试")
    print(
        f"监控文件: {(Path(__file__).resolve().parent.parent / 'config' / 'application.yml')}")
    print("请编辑 config/application.yml，随意添加键值对")
    print("支持中文键名、逗号分隔、不用缩进")
    print("修改后保存，程序将在 3 秒内自动检测并重新加载")
    print("=" * 60)
    print()

    iteration = 0
    while True:
        iteration += 1

        # ===== 随便读，键名跟 YAML 里写的一模一样就行 =====
        app_name = config.get('app_name', '未命名应用')
        start = config.get('数据起始日期', '2020-01-01')
        end = config.get('数据结束日期', '2025-12-31')
        fee = config.get('手续费率', 0.001)
        workers = config.get('最大并发数', 4)

        # 读取逗号分隔的列表，转成 Python 列表
        etf_raw = config.get('ETF池', 'sh.510300')
        etf_list = [x.strip() for x in etf_raw.split(',') if x.strip()]

        # 读取风控参数
        max_pos = config.get('风控-最大仓位%', 80)
        stop_loss = config.get('风控-止损线', 0.05)

        # 读取退出标志
        exit_flag = config.get('退出程序', False)

        # ===== 打印输出 =====
        print(f"--- 第 {iteration} 次读取 ---")
        print(f"  应用名称        : {app_name}")
        print(f"  数据区间        : {start} -> {end}")
        print(f"  手续费率        : {fee}")
        print(f"  并发数          : {workers}")
        print(f"  ETF池           : {etf_list}")
        print(f"  最大仓位        : {max_pos}%")
        print(f"  止损线          : {stop_loss}")
        print(f"  退出标志        : {exit_flag}")
        print("-" * 60)

        if exit_flag is True:
            print("检测到 退出程序: true，即将退出。")
            break

        time.sleep(3)

    print(" 测试结束")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n 用户中断")
