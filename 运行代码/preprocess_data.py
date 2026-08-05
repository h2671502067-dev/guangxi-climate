"""
广西气象数据预处理模块
=====================
将 data/ 原始数据源合并、清洗并落盘为预处理结果, 输出到 output/preprocessed/,
与原始数据(data/)分离, 作为后续分析的正式数据源。

输出文件:
- guangxi_station_features.csv   站点级38维特征 (8站)
- guangxi_province_average.csv   全省平均标准分析Schema

处理流程 (与 guangxi_data_loader / data_loader 逻辑保持一致):
1. 合并5个数据源 -> 站点级38维特征 (含月份周期编码)
2. Z-Score异常值剔除 (3σ, 置NaN后重新插值)
3. 8站逐月平均 + 标准Schema映射 (供步骤1-8分析)

用法:
    python preprocess_data.py
"""

import os
import numpy as np
import pandas as pd

from config import (PREPROCESSED_DIR, STATION_FEATURES_CSV,
                    PROVINCE_AVG_CSV, GUANGXI_STATIONS)
from guangxi_data_loader import load_guangxi_station_dfs
from data_loader import load_guangxi_avg


def build_station_features():
    """生成站点级38维特征CSV (8站)。

    强制 use_preprocessed=False, 从 data/ 原始数据现场合并清洗,
    避免读到上一次的预处理结果造成"自己抄自己"。
    """
    station_dfs = load_guangxi_station_dfs(use_preprocessed=False)
    frames = [station_dfs[s].copy() for s in GUANGXI_STATIONS]
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(STATION_FEATURES_CSV, index=False, encoding='utf-8-sig')
    print(f"  [1/2] 站点级38维特征: {out.shape[0]}行 x {out.shape[1]}列 "
          f"({len(GUANGXI_STATIONS)}站) -> {os.path.basename(STATION_FEATURES_CSV)}")
    return out


def build_province_average():
    """生成全省平均标准Schema CSV。

    强制 use_preprocessed=False, 从 data/ 原始数据现场计算,
    避免读到上一次的预处理结果。
    """
    df = load_guangxi_avg(use_preprocessed=False)
    df.to_csv(PROVINCE_AVG_CSV, index=False, encoding='utf-8-sig')
    print(f"  [2/2] 全省平均标准Schema: {df.shape[0]}行 x {df.shape[1]}列 "
          f"-> {os.path.basename(PROVINCE_AVG_CSV)}")
    return df


def run():
    """执行完整预处理, 生成 output/preprocessed/ 下全部CSV。"""
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    print("=" * 60)
    print("数据预处理: data/ 原始数据 -> output/preprocessed/")
    print("=" * 60)
    build_station_features()
    build_province_average()
    print("  预处理完成。\n")
    return True


if __name__ == '__main__':
    run()
