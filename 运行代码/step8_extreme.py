"""
步骤8: 极端事件与专题分析
==========================
包括极端事件检测、风速风向分析、土壤温湿度分析、辐射日照分析等。

数学方法:
  8.1 极端事件检测 (百分位法):
    极端高温阈值: P95 = 第95百分位数
    极端低温阈值: P05 = 第5百分位数
    当观测值 ≥ P95 或 ≤ P05 时, 判定为极端事件。

  8.2 Z-score异常值检测:
    Z = (x - μ) / σ
    |Z| > 2 时判定为异常值 (约占总体的4.6%)
    |Z| > 3 时判定为极端异常值 (约占总体的0.3%)

  8.3 风向统计:
    将360°方位角划分为8个方向:
      N(337.5°-22.5°), NE(22.5°-67.5°), E(67.5°-112.5°), ...
    统计各方向出现频率。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from config import CHART_DIR, SEASON_ORDER, SEASON_COLORS, plt_config

for k, v in plt_config.items():
    plt.rcParams[k] = v


def run(df, results):
    print("=" * 60)
    print("步骤8: 极端事件与专题分析")
    print("=" * 60)

    # ==================== 8.1 极端事件检测 ====================
    print("  [8.1] 极端事件检测 (百分位法)...")
    extremes = {}
    extreme_items = [
        ('最高气温(°C)', '极端高温', 0.95),
        ('最低气温(°C)', '极端低温', 0.05),
        ('总降水量(mm)', '极端降水', 0.95),
    ]
    if '最大阵风(km/h)' in df.columns:
        extreme_items.append(('最大阵风(km/h)', '极端大风', 0.95))
    for col, label, q in extreme_items:
        if col not in df.columns:
            continue
        threshold = df[col].quantile(q)
        if q == 0.05:
            extreme = df[df[col] <= threshold].sort_values(col)
        else:
            extreme = df[df[col] >= threshold].sort_values(col, ascending=False)
        extremes[label] = {'col': col, 'threshold': threshold, 'data': extreme}
        print(f"    {label} ({col}): 阈值={threshold:.1f}, 事件数={len(extreme)}")

    # ==================== 8.2 Z-score异常值检测 ====================
    print("  [8.2] Z-score异常值检测...")
    z_cols = [
        ('平均气温(°C)', '#E74C3C'), ('总降水量(mm)', '#3498DB'),
        ('参考蒸散量ET₀(mm)', '#F39C12')
    ]
    if '最大阵风(km/h)' in df.columns:
        z_cols.append(('最大阵风(km/h)', '#2ECC71'))
    z_results = {}
    for col, color in z_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        n_outlier = (z_scores > 2).sum()
        z_results[col] = z_scores
        print(f"    {col}: |Z|>2的异常值 {n_outlier} 个")

    # ==================== 8.3 可视化: 极端事件 ====================
    print("  [8.3] 绘制极端事件分析图...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, (col, color) in enumerate(z_cols):
        ax = axes[idx // 2, idx % 2]
        z_scores = z_results[col]
        colors_z = [color if z > 2 else '#333' for z in z_scores]
        ax.scatter(df['日期'], df[col], c=colors_z, s=15, alpha=0.7)
        mean_val = df[col].mean()
        std_val = df[col].std()
        ax.axhline(mean_val + 2 * std_val, color='r', linestyle='--', alpha=0.5, label='μ+2σ')
        ax.axhline(mean_val - 2 * std_val, color='r', linestyle='--', alpha=0.5, label='μ-2σ')
        ax.axhline(mean_val, color='gray', linestyle='-', alpha=0.3, label='μ')
        ax.set_title(f'{col} Z-score异常值检测', fontsize=12, fontweight='bold')
        ax.set_ylabel(col.split('(')[0])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('广西极端气象事件检测', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/19_极端事件检测.png')
    plt.close()

    # ==================== 8.4 风速风向分析 ====================
    print("  [8.4] 绘制风速风向分析图...")
    if '平均风速(km/h)' not in df.columns:
        print("    [警告] 缺少风速相关列, 跳过风速风向分析")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 各季节月均风速
        ax = axes[0, 0]
        for season in SEASON_ORDER:
            sdata = df[df['季节'] == season]
            sdata_sorted = sdata.sort_values('月份_int')
            ax.plot(sdata_sorted['月份_int'].values, sdata_sorted['平均风速(km/h)'].values,
                    'o-', color=SEASON_COLORS[season], label=season, alpha=0.5, markersize=3)
        ax.set_title('各季节月均风速', fontsize=12, fontweight='bold')
        ax.set_xlabel('月份')
        ax.set_ylabel('风速 (km/h)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 风速频率分布
        ax = axes[0, 1]
        ax.hist(df['平均风速(km/h)'], bins=30, color='#2ECC71', alpha=0.7, edgecolor='white')
        ax.axvline(df['平均风速(km/h)'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'均值={df["平均风速(km/h)"].mean():.1f}')
        ax.set_title('月均风速频率分布', fontsize=12, fontweight='bold')
        ax.set_xlabel('风速 (km/h)')
        ax.set_ylabel('频次')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 阵风 vs 平均风速
        ax = axes[1, 0]
        if '最大阵风(km/h)' in df.columns:
            ax.scatter(df['平均风速(km/h)'], df['最大阵风(km/h)'], alpha=0.5, s=20, c='#9B59B6')
            r = np.corrcoef(df['平均风速(km/h)'], df['最大阵风(km/h)'])[0, 1]
            z = np.polyfit(df['平均风速(km/h)'], df['最大阵风(km/h)'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df['平均风速(km/h)'].min(), df['平均风速(km/h)'].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'R²={r**2:.3f}')
            ax.set_xlabel('平均风速 (km/h)')
            ax.set_ylabel('最大阵风 (km/h)')
            ax.set_title('平均风速与最大阵风关系', fontsize=12, fontweight='bold')
            ax.legend()
        else:
            ax.text(0.5, 0.5, '最大阵风数据不可用', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('平均风速与最大阵风关系', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 风向分布
        ax = axes[1, 1]
        if '盛行风向(°)' in df.columns:
            wind_dir = df['盛行风向(°)'].dropna()
            # 将角度统一到 [0, 360) 范围
            wind_dir_wrapped = wind_dir % 360
            directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            # 每个方向中心角: 0, 45, 90, 135, 180, 225, 270, 315
            # 每个方向范围: [center-22.5, center+22.5)
            dir_counts = []
            for center in [0, 45, 90, 135, 180, 225, 270, 315]:
                low = (center - 22.5) % 360
                high = (center + 22.5) % 360
                if low < high:
                    count = ((wind_dir_wrapped >= low) & (wind_dir_wrapped < high)).sum()
                else:
                    count = ((wind_dir_wrapped >= low) | (wind_dir_wrapped < high)).sum()
                dir_counts.append(count)
            ax.bar(directions, dir_counts, color=plt.cm.hsv(np.linspace(0, 0.8, 8)), alpha=0.8, edgecolor='white')
            ax.set_title('盛行风向频率分布', fontsize=12, fontweight='bold')
            ax.set_ylabel('月份数')
        else:
            ax.text(0.5, 0.5, '盛行风向数据不可用', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('盛行风向频率分布', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle('广西风速与风向分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{CHART_DIR}/20_风速风向分析.png')
        plt.close()

    # ==================== 8.5 土壤温湿度分析 ====================
    print("  [8.5] 绘制土壤温湿度分析图...")
    soil_temp = ['土壤温度0-7cm(°C)']
    if '土壤温度7-28cm(°C)' in df.columns:
        soil_temp.append('土壤温度7-28cm(°C)')
    if '土壤温度28-100cm(°C)' in df.columns:
        soil_temp.append('土壤温度28-100cm(°C)')
    soil_moist = ['土壤湿度0-7cm(m³/m³)']
    if '土壤湿度7-28cm(m³/m³)' in df.columns:
        soil_moist.append('土壤湿度7-28cm(m³/m³)')
    if '土壤湿度28-100cm(m³/m³)' in df.columns:
        soil_moist.append('土壤湿度28-100cm(m³/m³)')
    soil_colors = ['#E74C3C', '#F39C12', '#3498DB']

    if not any(c in df.columns for c in soil_temp) or not any(c in df.columns for c in soil_moist):
        print("    [警告] 缺少土壤温湿度列, 跳过土壤温湿度分析")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        ax = axes[0, 0]
        for col, color in zip(soil_temp, soil_colors):
            if col not in df.columns:
                continue
            ax.plot(df['日期'], df[col], linewidth=0.8, color=color,
                    label=col.split('(')[0], alpha=0.8)
        ax.set_title('不同深度土壤温度变化', fontsize=12, fontweight='bold')
        ax.set_ylabel('温度 (°C)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        for col, color in zip(soil_moist, soil_colors):
            if col not in df.columns:
                continue
            ax.plot(df['日期'], df[col], linewidth=0.8, color=color,
                    label=col.split('(')[0], alpha=0.8)
        ax.set_title('不同深度土壤湿度变化', fontsize=12, fontweight='bold')
        ax.set_ylabel('湿度 (m³/m³)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if '土壤温度0-7cm(°C)' in df.columns:
            r = np.corrcoef(df['平均气温(°C)'], df['土壤温度0-7cm(°C)'])[0, 1]
            ax.scatter(df['平均气温(°C)'], df['土壤温度0-7cm(°C)'], alpha=0.5, s=20, c='#E74C3C')
            ax.set_title(f'气温与浅层土壤温度 (r={r:.3f})', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '土壤温度数据不可用', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('气温与浅层土壤温度', fontsize=12, fontweight='bold')
        ax.set_xlabel('气温 (°C)')
        ax.set_ylabel('土壤温度 (°C)')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if '土壤湿度0-7cm(m³/m³)' in df.columns:
            r = np.corrcoef(df['总降水量(mm)'], df['土壤湿度0-7cm(m³/m³)'])[0, 1]
            ax.scatter(df['总降水量(mm)'], df['土壤湿度0-7cm(m³/m³)'], alpha=0.5, s=20, c='#3498DB')
            ax.set_title(f'降水与浅层土壤湿度 (r={r:.3f})', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '土壤湿度数据不可用', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('降水与浅层土壤湿度', fontsize=12, fontweight='bold')
        ax.set_xlabel('降水量 (mm)')
        ax.set_ylabel('土壤湿度 (m³/m³)')
        ax.grid(True, alpha=0.3)

        fig.suptitle('广西土壤温湿度特征分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{CHART_DIR}/21_土壤温湿度分析.png')
        plt.close()

    # ==================== 8.6 辐射日照分析 ====================
    print("  [8.6] 绘制辐射日照分析图...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.plot(df['日期'], df['日照时数(h)'], color='#E67E22', linewidth=0.8)
    ax.set_title('日照时数月度变化', fontsize=12, fontweight='bold')
    ax.set_ylabel('日照时数 (h)')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if '平均云量(%)' in df.columns:
        r = np.corrcoef(df['平均云量(%)'], df['日照时数(h)'])[0, 1]
        ax.scatter(df['平均云量(%)'], df['日照时数(h)'], alpha=0.5, s=20, c='#3498DB')
        ax.set_xlabel('云量 (%)')
        ax.set_ylabel('日照时数 (h)')
        ax.set_title(f'云量与日照关系 (r={r:.3f})', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, '云量数据不可用', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('云量与日照关系', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(df['日期'], df['参考蒸散量ET₀(mm)'], color='#E74C3C', linewidth=0.8)
    ax.set_title('参考蒸散量ET₀月度变化', fontsize=12, fontweight='bold')
    ax.set_ylabel('ET₀ (mm)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    r = np.corrcoef(df['平均气温(°C)'], df['参考蒸散量ET₀(mm)'])[0, 1]
    sc = ax.scatter(df['平均气温(°C)'], df['参考蒸散量ET₀(mm)'], c=df['月份_int'],
                    cmap='RdYlBu_r', alpha=0.6, s=20)
    ax.set_xlabel('气温 (°C)')
    ax.set_ylabel('ET₀ (mm)')
    ax.set_title(f'气温与蒸散量关系 (r={r:.3f})', fontsize=12, fontweight='bold')
    plt.colorbar(sc, ax=ax, label='月份')
    ax.grid(True, alpha=0.3)

    fig.suptitle('广西辐射日照与蒸散量分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/22_辐射日照蒸散分析.png')
    plt.close()

    # ==================== 8.7 天气现象统计 ====================
    print("  [8.7] 绘制天气现象统计图...")
    if '天气现象' not in df.columns:
        print("    [警告] 缺少'天气现象'列, 跳过天气现象统计")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        weather_counts = df['天气现象'].value_counts()

        ax = axes[0]
        ax.pie(weather_counts.values, labels=weather_counts.index,
               autopct='%1.1f%%', colors=plt.cm.Set3(np.linspace(0, 1, len(weather_counts))),
               startangle=90)
        ax.set_title('天气现象总体分布', fontsize=13, fontweight='bold')

        ax = axes[1]
        weather_month = pd.crosstab(df['月份_int'], df['天气现象'])
        weather_month.plot(kind='bar', stacked=True, ax=ax, colormap='Set3', edgecolor='white')
        ax.set_xlabel('月份')
        ax.set_ylabel('月份数')
        ax.set_title('各月天气现象分布', fontsize=13, fontweight='bold')
        ax.legend(title='天气现象', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle('广西天气现象统计分析', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{CHART_DIR}/23_天气现象统计.png')
        plt.close()

    # ==================== 8.8 湿度分析 ====================
    print("  [8.8] 绘制湿度分析图...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.plot(df['日期'], df['平均湿度(%)'], color='#3498DB', linewidth=0.8, label='平均湿度')
    if '平均露点温度(°C)' in df.columns:
        ax.plot(df['日期'], df['平均露点温度(°C)'], color='#2ECC71', linewidth=0.8, label='露点温度')
    ax.set_title('湿度与露点温度月度变化', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    r = np.corrcoef(df['平均气温(°C)'], df['平均湿度(%)'])[0, 1]
    ax.scatter(df['平均气温(°C)'], df['平均湿度(%)'], alpha=0.5, s=20, c='#3498DB')
    ax.set_xlabel('气温 (°C)')
    ax.set_ylabel('湿度 (%)')
    ax.set_title(f'气温与湿度关系 (r={r:.3f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if '平均露点温度(°C)' in df.columns:
        r = np.corrcoef(df['平均湿度(%)'], df['平均露点温度(°C)'])[0, 1]
        ax.scatter(df['平均湿度(%)'], df['平均露点温度(°C)'], alpha=0.5, s=20, c='#2ECC71')
        ax.set_xlabel('湿度 (%)')
        ax.set_ylabel('露点温度 (°C)')
        ax.set_title(f'湿度与露点温度关系 (r={r:.3f})', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, '露点温度数据不可用', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('湿度与露点温度关系', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    month_humid = df.groupby('月份_int')['平均湿度(%)'].agg(['mean', 'std'])
    ax.bar(range(1, 13), month_humid['mean'], yerr=month_humid['std'],
           color='#3498DB', alpha=0.6, capsize=3)
    ax.set_xlabel('月份')
    ax.set_ylabel('湿度 (%)')
    ax.set_title('各月平均湿度(含标准差)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('广西湿度特征分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/24_湿度分析.png')
    plt.close()

    # ==================== 8.9 气压分析 ====================
    print("  [8.9] 绘制气压分析图...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    ax.plot(df['日期'], df['平均海平面气压(hPa)'], color='#8E44AD', linewidth=0.8)
    ax.set_title('海平面气压月度变化', fontsize=12, fontweight='bold')
    ax.set_ylabel('气压 (hPa)')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for season in SEASON_ORDER:
        sdata = df[df['季节'] == season]
        sdata_sorted = sdata.sort_values('月份_int')
        ax.plot(sdata_sorted['月份_int'].values, sdata_sorted['平均海平面气压(hPa)'].values,
                'o-', color=SEASON_COLORS[season], label=season, alpha=0.5, markersize=3)
    ax.set_title('各季节月均气压', fontsize=12, fontweight='bold')
    ax.set_xlabel('月份')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    r = np.corrcoef(df['平均气温(°C)'], df['平均海平面气压(hPa)'])[0, 1]
    sc = ax.scatter(df['平均气温(°C)'], df['平均海平面气压(hPa)'],
                    alpha=0.5, s=20, c=df['月份_int'], cmap='RdYlBu_r')
    ax.set_xlabel('气温 (°C)')
    ax.set_ylabel('气压 (hPa)')
    ax.set_title(f'气温与气压关系 (r={r:.3f})', fontsize=12, fontweight='bold')
    plt.colorbar(sc, ax=ax, label='月份')
    ax.grid(True, alpha=0.3)

    fig.suptitle('广西气压特征分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/25_气压分析.png')
    plt.close()

    # ==================== 8.10 气候特征雷达图 ====================
    print("  [8.10] 绘制气候特征雷达图...")
    yearly_features = df.groupby('年份_int').agg({
        '平均气温(°C)': 'mean', '总降水量(mm)': 'sum', '平均湿度(%)': 'mean',
        '平均风速(km/h)': 'mean', '日照时数(h)': 'sum', '参考蒸散量ET₀(mm)': 'sum'
    }).reset_index()

    all_years = sorted(df['年份_int'].unique())
    typical_years = [all_years[0], all_years[len(all_years)//4], all_years[len(all_years)//2],
                     all_years[3*len(all_years)//4], all_years[-1]]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    categories = ['年均气温', '年总降水', '平均湿度', '平均风速', '日照时数', '参考蒸散量']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    colors_radar = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    for idx, year in enumerate(typical_years):
        if year in yearly_features['年份_int'].values:
            row = yearly_features[yearly_features['年份_int'] == year].iloc[0]
            vals = [row['平均气温(°C)'], row['总降水量(mm)'], row['平均湿度(%)'],
                    row['平均风速(km/h)'], row['日照时数(h)'], row['参考蒸散量ET₀(mm)']]
            vals_norm = []
            for i, v in enumerate(vals):
                vmin = yearly_features.iloc[:, i + 1].min()
                vmax = yearly_features.iloc[:, i + 1].max()
                vals_norm.append((v - vmin) / (vmax - vmin) if vmax > vmin else 0.5)
            vals_norm += vals_norm[:1]
            ax.plot(angles, vals_norm, 'o-', linewidth=2, label=str(year), color=colors_radar[idx])
            ax.fill(angles, vals_norm, alpha=0.1, color=colors_radar[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title('广西典型年份气候特征雷达图', fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/26_气候特征雷达图.png')
    plt.close()

    results['extremes'] = extremes
    results['z_results'] = {col: {'n_outlier': int((z > 2).sum())} for col, z in z_results.items()}
    print("  步骤8完成, 共生成8张图表。\n")
    return results
