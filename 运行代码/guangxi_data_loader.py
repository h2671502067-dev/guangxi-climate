"""
广西气象多源数据加载模块
=========================
合并5个数据源为38维多变量特征集:
- 地面气象 (NASA POWER): 24个特征
- 干旱指数: 4个特征 (SPI/SPEI)
- 高空衍生 (NCEP): 7个特征
- 植被指数 NDVI (MODIS): 1个特征
- 月份周期编码: 2个特征 (month_sin/month_cos)

总计38维特征, 预测目标: 气温_2m_℃ (2m气温)
"""

import os
import numpy as np
import pandas as pd

from config import (GUANGXI_DATA_DIR, GUANGXI_STATIONS, GUANGXI_TARGET,
                    GUANGXI_FEATURES_GROUND, GUANGXI_FEATURES_DROUGHT,
                    GUANGXI_FEATURES_UPPER, GUANGXI_NDVI_COL,
                    GUANGXI_FEATURES, STATION_FEATURES_CSV)


def _load_csv(filename):
    """读取CSV并跳过#注释行"""
    path = os.path.join(GUANGXI_DATA_DIR, filename)
    return pd.read_csv(path, encoding='utf-8', comment='#')


def _replace_outliers_zscore(df, features, threshold=3.0):
    """Z-Score异常值检测: 将|z|>threshold的样本置为NaN。

    时序数据直接删行会破坏连续性, 故采用"置NaN+后续插值"策略,
    既剔除极端值影响又保留时间轴完整。返回被替换的异常值总数。
    """
    n_outliers = 0
    for col in features:
        std = df[col].std()
        if std == 0 or pd.isna(std):
            continue
        z = (df[col] - df[col].mean()) / std
        mask = z.abs() > threshold
        n = int(mask.sum())
        if n:
            df.loc[mask, col] = np.nan
            n_outliers += n
    return n_outliers


def load_guangxi_station_dfs(stations=None, use_preprocessed=True):
    """合并5个数据源, 返回 {station: DataFrame(38维特征+日期列)}。

    优先读取 output/preprocessed/ 中已生成的站点级特征CSV (预处理结果),
    未生成时回退到原始数据现场处理。

    Parameters
    ----------
    stations : list or None
        指定站点列表, 默认使用全部8个代表性城市。
    use_preprocessed : bool
        是否优先使用预处理CSV, 默认True。

    Returns
    -------
    dict[str, pd.DataFrame]
        键为站点英文名, 值为按时间升序排列的38维特征DataFrame。
    """
    stations = stations or GUANGXI_STATIONS

    # 优先使用预处理结果
    if use_preprocessed and os.path.exists(STATION_FEATURES_CSV):
        merged = pd.read_csv(STATION_FEATURES_CSV, encoding='utf-8-sig')
        merged['日期'] = pd.to_datetime(merged['日期'])
        result = {}
        for s in stations:
            sub = merged[merged['station'] == s].copy().reset_index(drop=True)
            result[s] = sub
        return result

    ground = _load_csv('guangxi_ground_meteorology_monthly.csv')
    drought = _load_csv('guangxi_drought_indices_monthly.csv')
    ndvi = _load_csv('guangxi_ndvi_monthly.csv')
    upper = _load_csv('guangxi_upper_air_derived_monthly.csv')

    keys = ['station', 'year', 'month']

    # 逐步左连接合并 (保留经纬度; 晴空短波辐射供data_loader日照估算用)
    merged = ground[keys + ['longitude', 'latitude'] + GUANGXI_FEATURES_GROUND +
                    ['短波辐射_晴空_kWhm2day']]
    merged = merged.merge(drought[keys + GUANGXI_FEATURES_DROUGHT], on=keys, how='left')
    merged = merged.merge(ndvi[keys + [GUANGXI_NDVI_COL]], on=keys, how='left')
    merged = merged.merge(upper[keys + GUANGXI_FEATURES_UPPER], on=keys, how='left')

    # 月份周期编码 (气象数据强季节性)
    merged['month_sin'] = np.sin(2 * np.pi * merged['month'] / 12)
    merged['month_cos'] = np.cos(2 * np.pi * merged['month'] / 12)

    # 日期与排序
    merged['日期'] = pd.to_datetime(
        merged['year'].astype(int).astype(str) + '-' +
        merged['month'].astype(int).astype(str) + '-01'
    )
    merged = merged.sort_values(['station', '日期']).reset_index(drop=True)

    result = {}
    total_outliers = 0
    for s in stations:
        sub = merged[merged['station'] == s].copy()
        # 缺失值处理: 线性插值 + 边界填充
        sub[GUANGXI_FEATURES] = sub[GUANGXI_FEATURES].interpolate(method='linear')
        sub[GUANGXI_FEATURES] = sub[GUANGXI_FEATURES].bfill().ffill()
        # Z-Score异常值剔除: 置NaN后重新插值, 保留时序连续性
        n_out = _replace_outliers_zscore(sub, GUANGXI_FEATURES, threshold=3.0)
        sub[GUANGXI_FEATURES] = sub[GUANGXI_FEATURES].interpolate(method='linear').bfill().ffill()
        total_outliers += n_out
        result[s] = sub.reset_index(drop=True)

    if total_outliers:
        print(f"  Z-Score异常值处理(阈值3σ): 共剔除 {total_outliers} 个极端值")
    return result


if __name__ == '__main__':
    dfs = load_guangxi_station_dfs()
    for s, df in dfs.items():
        print(f"{s}: {len(df)}行 x {df.shape[1]}列, "
              f"时间 {df['日期'].min().date()} ~ {df['日期'].max().date()}, "
              f"特征数 = {len(GUANGXI_FEATURES)}")
    df0 = dfs[list(dfs.keys())[0]]
    print(f"\n特征维度: {len(GUANGXI_FEATURES)} (期望38)")
    print(f"示例列: {list(df0.columns[:8])}")
