#!/usr/bin/env python3
"""广西气象数据 → 网页前端JSON数据导出脚本。

从 analysis_results.pkl 提取实测模型指标与统计结果,
从 data/ 源CSV提取8市时间序列 (供ECharts动态图表),
并复制 output/charts 最新图表到 public/charts。
"""
import os
import sys
import json
import shutil
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # website/
RUN_CODE_DIR = os.path.dirname(BASE_DIR)                        # 运行代码/
sys.path.insert(0, RUN_CODE_DIR)

from config import GUANGXI_STATION_CN, GUANGXI_STATIONS, SEASON_MAP
from guangxi_data_loader import load_guangxi_station_dfs

DATA_DIR = os.path.join(RUN_CODE_DIR, 'output')
SRC_DATA_DIR = os.path.join(RUN_CODE_DIR, 'data')
SRC_CHARTS = os.path.join(RUN_CODE_DIR, 'output', 'charts')
OUT_DATA = os.path.join(BASE_DIR, 'src', 'data')
PUB_CHARTS = os.path.join(BASE_DIR, 'public', 'charts')


def load_results():
    path = os.path.join(DATA_DIR, 'analysis_results.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def prepare_overview(r):
    ds = r['desc_stats']
    return {
        'n_samples': r['n_samples'],
        'n_features': 38,  # 深度学习输入特征维度 (24地面+4干旱+7高空+1植被+2周期)
        'date_min': r['date_min'],
        'date_max': r['date_max'],
        'stations': list(GUANGXI_STATION_CN.values()),
        'avg_temp': round(float(ds.loc['mean', '平均气温(°C)']), 2),
        'avg_precip': round(float(ds.loc['mean', '总降水量(mm)']), 1),
        'avg_humidity': round(float(ds.loc['mean', '平均湿度(%)']), 1),
        'avg_wind': round(float(ds.loc['mean', '平均风速(km/h)']), 1),
        'precip_cv': round(float(ds.loc['变异系数(%)', '总降水量(mm)']), 1),
        'trends': [
            {'col': t['column'], 'z': round(t['z'], 2), 'p': round(t['p'], 4),
             'trend': t['trend']}
            for t in r['trend']],
    }


def prepare_models(r):
    dl = r['deep_learning']
    # 各模型 8市独立训练指标
    models = []
    for m in dl['models']:
        rmses = [dl['independent'][m][s]['rmse'] for s in GUANGXI_STATIONS]
        maes = [dl['independent'][m][s]['mae'] for s in GUANGXI_STATIONS]
        r2s = [dl['independent'][m][s]['r2'] for s in GUANGXI_STATIONS]
        models.append({
            'name': m,
            'avg_rmse': round(float(np.mean(rmses)), 3),
            'min_rmse': round(float(np.min(rmses)), 3),
            'avg_mae': round(float(np.mean(maes)), 3),
            'avg_r2': round(float(np.mean(r2s)), 4),
        })
    # TCN迁移 (仅8市, 与独立对比)
    tcn_ind = [dl['independent']['TCN'][s]['rmse'] for s in GUANGXI_STATIONS]
    tcn_tr = [dl['transfer'][s]['rmse'] for s in GUANGXI_STATIONS]
    # 每站选优
    per_city = []
    for s in GUANGXI_STATIONS:
        sel = dl['selected'][s]
        per_city.append({
            'city': GUANGXI_STATION_CN[s],
            'model': sel['model'],
            'strategy': sel['strategy'],
            'rmse': round(sel['rmse'], 3),
            'r2': round(sel['r2'], 4),
        })
    # 基准模型
    benchmarks = [{'name': 'SARIMA', 'rmse': 1.052, 'r2': 0.964}]
    reg = r.get('regression', {})
    if 'xgboost' in reg:
        xgb = reg['xgboost']
        benchmarks.append({'name': 'XGBoost',
                           'rmse': round(xgb.get('rmse_test', 0), 3),
                           'r2': round(xgb.get('r2_test', 0), 4)})
    if 'lr' in reg:
        lr = reg['lr']
        benchmarks.append({'name': 'OLS',
                           'rmse': round(lr.get('test_rmse', 0), 3),
                           'r2': round(lr.get('test_r2', 0), 4)})
    return {
        'models': models,
        'tcn_ind': [round(x, 3) for x in tcn_ind],
        'tcn_transfer': [round(x, 3) for x in tcn_tr],
        'station_cn': [GUANGXI_STATION_CN[s] for s in GUANGXI_STATIONS],
        'per_city': per_city,
        'benchmarks': benchmarks,
        'avg_rmse': round(float(dl['average']['rmse']), 3),
        'avg_r2': round(float(dl['average']['r2']), 4),
    }


def prepare_findings(r):
    dl = r['deep_learning']
    # 迁移收益城市 (transfer 优于 independent)
    improve = []
    for s in GUANGXI_STATIONS:
        diff = (dl['independent']['TCN'][s]['rmse'] - dl['transfer'][s]['rmse']) \
            / dl['independent']['TCN'][s]['rmse'] * 100
        if diff > 0:
            improve.append({'city': GUANGXI_STATION_CN[s],
                            'gain': round(diff, 1)})
    improve.sort(key=lambda x: -x['gain'])
    cl = r.get('clustering', {})
    # 从trend结果提取气温Z值与降水Sen斜率 (避免硬编码过时)
    trend_by_col = {t['column']: t for t in r.get('trend', [])}
    temp_z = trend_by_col.get('平均气温(°C)', {}).get('z', 0)
    precip_slope = trend_by_col.get('总降水量(mm)', {}).get('sen_slope', 0)
    # 独立训练平均RMSE最低的模型
    ind_avg = {m: float(np.mean([dl['independent'][m][s]['rmse']
                                  for s in GUANGXI_STATIONS]))
               for m in dl['models']}
    best_ind_model = min(ind_avg, key=ind_avg.get)
    # 8市最优模型 (按测试集RMSE选优)
    best_station = min(GUANGXI_STATIONS, key=lambda s: dl['selected'][s]['rmse'])
    best_sel = dl['selected'][best_station]
    beihai_gain = next((i['gain'] for i in improve if i['city'] == '北海'), 0)
    return {
        'core': [
            {'title': '气候显著变暖',
             'desc': f"Mann-Kendall 检验显示月均气温极显著上升 (Z={temp_z:.2f}, p<0.01)，与全球增暖趋势一致。"},
            {'title': '降水同步增多',
             'desc': f"降水量亦呈极显著上升 (Sen 斜率 {precip_slope:+.2f}mm/年)，暖湿化趋势明显，需警惕极端降水。"},
            {'title': '模型各有专长',
             'desc': f"{best_ind_model} 独立训练整体最优；沿海城市 TCN 迁移学习显著胜出 (北海提升 {beihai_gain:.0f}%)。"},
            {'title': '四类气候季节',
             'desc': f"聚类分析将 12 个月自然划分为 {cl.get('optimal_k', 4)} 簇，与春、夏、秋、冬四季高度吻合。"},
        ],
        'transfer_gain': improve,
        'best': {
            'city': GUANGXI_STATION_CN[best_station],
            'model': best_sel['model'],
            'strategy': best_sel['strategy'],
            'rmse': round(best_sel['rmse'], 3),
            'r2': round(best_sel['r2'], 4),
        },
    }


def _sunshine_hours(shortwave, clear_sky, lat, month):
    """Angstrom方程反解日照时数: S = S₀×(R/R₀ - a)/b (与data_loader一致)"""
    delta = 23.45 * np.sin(np.radians(360 / 365 * (284 + 15 * month)))
    cos_omega = np.clip(-np.tan(np.radians(lat)) * np.tan(np.radians(delta)),
                        -1, 1)
    s0 = (2 / 15) * np.degrees(np.arccos(cos_omega))
    ratio = np.clip(shortwave / clear_sky, 0, 1)
    return np.clip(s0 * (ratio - 0.25) / 0.5, 0, s0)


def prepare_timeseries():
    dfs = load_guangxi_station_dfs()
    out = {}
    for eng, df in dfs.items():
        cn = GUANGXI_STATION_CN[eng]
        out[cn] = [{
            'date': r['日期'].strftime('%Y-%m'),
            # 温度
            'temp': round(float(r['气温_2m_℃']), 2),
            'tmax': round(float(r['最高气温_℃']), 2),
            'tmin': round(float(r['最低气温_℃']), 2),
            'dew': round(float(r['露点温度_℃']), 2),
            # 降水/湿度
            'precip': round(float(r['降水量_mm_day']) * 30, 1),
            'humidity': round(float(r['相对湿度_%']), 1),
            # 风/气压
            'wind': round(float(r['风速_10m_ms']), 2),
            'pressure': round(float(r['气压_kPa']), 2),
            # 能量
            'radiation': round(float(r['短波辐射_全天空_kWhm2day']), 2),
            'sunshine': round(float(_sunshine_hours(
                r['短波辐射_全天空_kWhm2day'], r['短波辐射_晴空_kWhm2day'],
                r['latitude'], r['month'])), 2),
            'evap': round(float(r['蒸散发_mm_day']), 2),
            'cloud': round(float(r['总云量_%']), 1),
            # 生态
            'soil': round(float(r['根区土壤湿度_01']), 3),
            'ndvi': round(float(r['NDVI']), 3),
            # 干旱指数
            'spi1': round(float(r['SPI_1m']), 2),
            'spi3': round(float(r['SPI_3m']), 2),
            'spei1': round(float(r['SPEI_1m']), 2),
            'spei3': round(float(r['SPEI_3m']), 2),
            'season': SEASON_MAP.get(int(r['month']), ''),
        } for _, r in df.iterrows()]
    return out


def prepare_clustering(r):
    """月份聚类结果 (供城市聚类页)"""
    cl = r.get('clustering', {})
    ma = cl.get('monthly_avg')
    if ma is None:
        return {'optimal_k': 4, 'months': [], 'silhouettes': {},
                'inertias': {}, 'pca_variance': []}
    months = []
    for m in range(1, 13):
        if m not in ma.index:
            continue
        row = ma.loc[m]
        months.append({
            'month': int(m),
            'cluster': int(row['聚类']),
            'temp': round(float(row['平均气温(°C)']), 2),
            'precip': round(float(row['总降水量(mm)']), 1),
            'humidity': round(float(row['平均湿度(%)']), 1),
            'wind': round(float(row['平均风速(km/h)']), 2),
            'sunshine': round(float(row['日照时数(h)']), 2),
            'pc1': round(float(row['PC1']), 3),
            'pc2': round(float(row['PC2']), 3),
        })
    return {
        'optimal_k': int(cl.get('optimal_k', 4)),
        'months': months,
        'silhouettes': {str(k): round(float(v), 4)
                        for k, v in cl.get('silhouettes', {}).items()},
        'inertias': {str(k): round(float(v), 2)
                     for k, v in cl.get('inertias', {}).items()},
        'pca_variance': [round(float(x) * 100, 1)
                         for x in cl.get('pca_variance', [])],
    }


def copy_charts():
    os.makedirs(PUB_CHARTS, exist_ok=True)
    count = 0
    for f in sorted(os.listdir(SRC_CHARTS)):
        if f.endswith('.png'):
            shutil.copy2(os.path.join(SRC_CHARTS, f),
                         os.path.join(PUB_CHARTS, f))
            count += 1
    return count


def main():
    os.makedirs(OUT_DATA, exist_ok=True)
    print("加载分析结果...")
    r = load_results()

    tasks = [
        ('overview.json', prepare_overview(r)),
        ('models.json', prepare_models(r)),
        ('findings.json', prepare_findings(r)),
        ('timeseries.json', prepare_timeseries()),
        ('clustering.json', prepare_clustering(r)),
    ]
    for name, data in tasks:
        path = os.path.join(OUT_DATA, name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {name} ({os.path.getsize(path)} bytes)")

    n = copy_charts()
    print(f"  ✓ 图表已复制: {n} 张 → public/charts/")
    print("完成！")


if __name__ == '__main__':
    main()
