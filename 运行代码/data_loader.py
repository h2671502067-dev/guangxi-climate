"""
模块1: 数据加载与清洗 (广西全省平均)
=====================================
加载广西8市气象数据并计算全省平均, 映射为标准分析Schema,
使步骤1-10共享同一套分析流程。

数据映射关系 (广西源列 -> 标准列):
  - 气温_2m_℃ -> 平均气温(°C)
  - 最高气温_℃ -> 最高气温(°C)
  - 最低气温_℃ -> 最低气温(°C)
  - 相对湿度_% -> 平均湿度(%)
  - 降水量_mm_day -> 总降水量(mm)     (×当月天数)
  - 风速_10m_ms  -> 平均风速(km/h)    (×3.6)
  - 短波辐射/晴空辐射 -> 日照时数(h)  (Angstrom方程反解)
  - 蒸散发_mm_day -> 参考蒸散量ET₀(mm) (×当月天数)
  - 气压_kPa     -> 平均海平面气压(hPa) (×10)
  - 地表温度_℃  -> 土壤温度0-7cm(°C)
  - 表层土壤湿度_01 -> 土壤湿度0-7cm(m³/m³)
"""

import os

import numpy as np
import pandas as pd

from config import (SEASON_MAP, CORE_COLUMNS,
                    GUANGXI_DATA_DIR, GUANGXI_STATIONS, GUANGXI_FEATURES,
                    PROVINCE_AVG_CSV)

GUANGXI_CSV = os.path.join(GUANGXI_DATA_DIR,
                           'guangxi_ground_meteorology_monthly.csv')


# ======================== 广西数据加载 ========================

def _sunshine_hours(shortwave, clear_sky, lat, month):
    """Angstrom方程反解日照时数: S = S₀×(R/R₀ - a)/b

    S₀: 理论日照时数(由纬度与太阳赤纬计算)
    R/R₀: 实际短波辐射/晴空短波辐射比, a=0.25, b=0.5
    """
    delta = 23.45 * np.sin(np.radians(360 / 365 * (284 + 15 * month)))
    cos_omega = np.clip(-np.tan(np.radians(lat)) * np.tan(np.radians(delta)),
                        -1, 1)
    s0 = (2 / 15) * np.degrees(np.arccos(cos_omega))
    ratio = np.clip(shortwave / clear_sky, 0, 1)
    return np.clip(s0 * (ratio - 0.25) / 0.5, 0, s0)


def load_guangxi_avg(use_preprocessed=True):
    """加载广西8站数据, 计算全省平均并映射为标准Schema。

    优先读取 output/preprocessed/ 中已生成的全省平均CSV (预处理结果),
    未生成时回退到原始数据现场计算。
    """
    # 优先使用预处理结果
    if use_preprocessed and os.path.exists(PROVINCE_AVG_CSV):
        df = pd.read_csv(PROVINCE_AVG_CSV, encoding='utf-8-sig')
        df['日期'] = pd.to_datetime(df['日期'])
        return df

    from guangxi_data_loader import load_guangxi_station_dfs

    station_dfs = load_guangxi_station_dfs()
    df = station_dfs[GUANGXI_STATIONS[0]].copy()

    # 8站逐月平均 (含纬度与晴空短波辐射)
    for f in GUANGXI_FEATURES + ['短波辐射_晴空_kWhm2day']:
        df[f] = np.mean([d[f].values for d in station_dfs.values()], axis=0)
    df['纬度'] = np.mean([d['latitude'].values for d in station_dfs.values()],
                         axis=0)

    df['年份'] = df['year'].astype(int)
    df['月份'] = df['month'].astype(int)
    days = df['日期'].dt.days_in_month

    # 列映射
    schema = {
        '平均气温(°C)': df['气温_2m_℃'],
        '最高气温(°C)': df['最高气温_℃'],
        '最低气温(°C)': df['最低气温_℃'],
        '平均湿度(%)': df['相对湿度_%'],
        '总降水量(mm)': df['降水量_mm_day'] * days,
        '平均风速(km/h)': df['风速_10m_ms'] * 3.6,
        '日照时数(h)': _sunshine_hours(
            df['短波辐射_全天空_kWhm2day'].values,
            df['短波辐射_晴空_kWhm2day'].values,
            df['纬度'].values, df['月份'].values),
        '参考蒸散量ET₀(mm)': df['蒸散发_mm_day'] * days,
        '平均海平面气压(hPa)': df['气压_kPa'] * 10,
        '土壤温度0-7cm(°C)': df['地表温度_℃'],
        '土壤湿度0-7cm(m³/m³)': df['表层土壤湿度_01'],
        # 扩展列 (步骤8专题分析使用)
        '最大阵风(km/h)': df['最大风速_10m_ms'] * 3.6,
        '盛行风向(°)': df['风向_10m_度'],
        '平均云量(%)': df['总云量_%'],
        '平均露点温度(°C)': df['露点温度_℃'],
    }
    out = pd.DataFrame(schema)
    out['日期'] = df['日期'].values
    out['年份_int'] = df['年份'].values
    out['月份_int'] = df['月份'].values
    out['季节'] = out['月份_int'].map(SEASON_MAP)
    return out


# ======================== 统一入口 ========================

def load_and_clean():
    """加载并清洗广西全省平均数据。"""
    print("=" * 60)
    print("模块1: 数据加载与清洗 (广西全省平均, 8市)")
    print("=" * 60)
    df = load_guangxi_avg()
    numeric_cols = [c for c in df.columns
                    if c not in ['日期', '年份_int', '月份_int', '季节']]
    invalid_cols = []
    print(f"  数据维度: {df.shape[0]}行 × {df.shape[1]}列")
    print(f"  时间范围: {df['日期'].min().strftime('%Y-%m')} ~ "
          f"{df['日期'].max().strftime('%Y-%m')}")
    print(f"  数据来源: 广西8市 (南宁/桂林/柳州/梧州/北海/百色/河池/防城港) 全省平均")
    print(f"  缺失值: 已通过插值+边界填充处理")
    print(f"  核心分析列数: {len(CORE_COLUMNS)}")
    print(f"  无效列: {invalid_cols}")
    print()
    return df, numeric_cols, invalid_cols
