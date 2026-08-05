"""
步骤1: 描述性统计分析
=====================
对核心气象指标进行全面的描述性统计, 包括集中趋势、离散程度、分布形态等。

数学方法:
  - 均值: x̄ = (1/n) Σ xᵢ
  - 标准差: s = √[(1/(n-1)) Σ (xᵢ - x̄)²]
  - 偏度: Skew = (n/((n-1)(n-2))) Σ ((xᵢ - x̄)/s)³
    Skew > 0 右偏, Skew < 0 左偏, Skew = 0 对称
  - 峰度: Kurt = [(n(n+1))/((n-1)(n-2)(n-3))] Σ ((xᵢ - x̄)/s)⁴
    - 3[(n-1)²/((n-2)(n-3))]
    Kurt > 0 尖峰, Kurt < 0 扁平
  - 变异系数: CV = (s / x̄) × 100%
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from config import CORE_COLUMNS, CHART_DIR, SEASON_ORDER, SEASON_COLORS, plt_config

for k, v in plt_config.items():
    plt.rcParams[k] = v


def run(df, results):
    print("=" * 60)
    print("步骤1: 描述性统计分析")
    print("=" * 60)

    # ==================== 1.1 统计摘要 ====================
    desc = df[CORE_COLUMNS].describe()
    desc.loc['偏度'] = df[CORE_COLUMNS].skew()
    desc.loc['峰度'] = df[CORE_COLUMNS].kurtosis()
    # [修复#17] CV计算: 防止mean=0时产生inf
    cv = pd.Series(index=df[CORE_COLUMNS].columns, dtype=float)
    for c in CORE_COLUMNS:
        mean_val = df[c].mean()
        if abs(mean_val) > 1e-10:
            cv[c] = (df[c].std() / mean_val * 100)
        else:
            cv[c] = np.nan
    desc.loc['变异系数(%)'] = cv

    print("  [1.1] 核心指标统计摘要:")
    print(desc.round(2).to_string())
    print()

    # ==================== 1.2 核心气象要素月度时间序列 ====================
    print("  [1.2] 绘制核心气象要素月度时间序列图...")
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)

    plot_specs = [
        ('平均气温(°C)', '最高气温(°C)', '最低气温(°C)', '气温变化 (°C)', '#E74C3C', '#F39C12', '#3498DB'),
        ('总降水量(mm)', None, None, '月降水量 (mm)', '#3498DB', None, None),
        ('平均湿度(%)', None, None, '月平均湿度 (%)', '#2ECC71', None, None),
        ('平均风速(km/h)', '最大阵风(km/h)', None, '风速变化 (km/h)', '#9B59B6', '#1ABC9C', None),
    ]
    # 过滤掉引用不存在列的plot_specs
    plot_specs = [
        spec for spec in plot_specs
        if spec[0] in df.columns and (spec[1] is None or spec[1] in df.columns)
        and (spec[2] is None or spec[2] in df.columns)
    ]
    for i, (c1, c2, c3, title, clr1, clr2, clr3) in enumerate(plot_specs):
        ax = axes[i]
        ax.plot(df['日期'], df[c1], color=clr1, linewidth=0.8, label=c1, alpha=0.9)
        if c2:
            ax.plot(df['日期'], df[c2], color=clr2, linewidth=0.6, label=c2, alpha=0.6)
        if c3:
            ax.plot(df['日期'], df[c3], color=clr3, linewidth=0.6, label=c3, alpha=0.6)
        ax.set_ylabel(title, fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight='bold')

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.suptitle('广西核心气象要素月度变化趋势 (2005-2025)', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/01_核心气象要素时间序列.png')
    plt.close()

    # ==================== 1.3 气温月度箱线图 ====================
    print("  [1.3] 绘制气温月度箱线图...")
    fig, ax = plt.subplots(figsize=(14, 6))
    month_data = [df[df['月份_int'] == m]['平均气温(°C)'].values for m in range(1, 13)]
    bp = ax.boxplot(month_data, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    ax.set_xticklabels([f'{m}月' for m in range(1, 13)])
    cmap = plt.cm.RdYlBu_r
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(cmap(i / 11))
    ax.set_xlabel('月份', fontsize=12)
    ax.set_ylabel('气温 (°C)', fontsize=12)
    ax.set_title('广西各月平均气温分布箱线图 (2005-2025)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/02_气温月度箱线图.png')
    plt.close()

    # ==================== 1.4 降水量月度箱线图 ====================
    print("  [1.4] 绘制降水量月度箱线图...")
    fig, ax = plt.subplots(figsize=(14, 6))
    precip_data = [df[df['月份_int'] == m]['总降水量(mm)'].values for m in range(1, 13)]
    bp = ax.boxplot(precip_data, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    ax.set_xticklabels([f'{m}月' for m in range(1, 13)])
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(plt.cm.Blues(0.3 + i * 0.06))
    ax.set_xlabel('月份', fontsize=12)
    ax.set_ylabel('降水量 (mm)', fontsize=12)
    ax.set_title('广西各月降水量分布箱线图 (2005-2025)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/03_降水量月度箱线图.png')
    plt.close()

    # ==================== 1.5 年度气象指标趋势 ====================
    print("  [1.5] 绘制年度气象指标趋势图...")
    yearly = df.groupby('年份_int').agg({
        '平均气温(°C)': 'mean', '最高气温(°C)': 'max', '最低气温(°C)': 'min',
        '总降水量(mm)': 'sum', '平均风速(km/h)': 'mean', '日照时数(h)': 'sum',
        '参考蒸散量ET₀(mm)': 'sum'
    }).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    bar_specs = [
        ('平均气温(°C)', '年均气温变化趋势', '气温 (°C)', '#E74C3C'),
        ('总降水量(mm)', '年总降水量变化趋势', '降水量 (mm)', '#3498DB'),
        ('平均风速(km/h)', '年均风速变化趋势', '风速 (km/h)', '#2ECC71'),
        ('日照时数(h)', '年日照时数变化趋势', '日照时数 (h)', '#F39C12'),
    ]
    for idx, (col, title, ylabel, color) in enumerate(bar_specs):
        ax = axes[idx // 2, idx % 2]
        ax.bar(yearly['年份_int'], yearly[col], color=color, alpha=0.6)
        z = np.polyfit(yearly['年份_int'], yearly[col], 1)
        p = np.poly1d(z)
        ax.plot(yearly['年份_int'], p(yearly['年份_int']), 'k--', linewidth=2,
                label=f'趋势线 (斜率={z[0]:+.3f})')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle('广西年度气象指标变化趋势 (2005-2025)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/04_年度气象指标趋势.png')
    plt.close()

    # ==================== 1.6 气温热力图 ====================
    print("  [1.6] 绘制气温热力图...")
    pivot_temp = df.pivot_table(values='平均气温(°C)', index='年份_int', columns='月份_int')
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot_temp, annot=True, fmt='.1f', cmap='RdYlBu_r', center=0,
                linewidths=0.5, ax=ax, cbar_kws={'label': '气温 (°C)'})
    ax.set_xlabel('月份', fontsize=12)
    ax.set_ylabel('年份', fontsize=12)
    ax.set_title('广西月平均气温热力图 (2005-2025)', fontsize=14, fontweight='bold')
    ax.set_xticklabels([f'{m}月' for m in range(1, 13)])
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/05_气温热力图.png')
    plt.close()

    # ==================== 1.7 降水热力图 ====================
    print("  [1.7] 绘制降水热力图...")
    pivot_precip = df.pivot_table(values='总降水量(mm)', index='年份_int', columns='月份_int')
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot_precip, annot=True, fmt='.0f', cmap='Blues',
                linewidths=0.5, ax=ax, cbar_kws={'label': '降水量 (mm)'})
    ax.set_xlabel('月份', fontsize=12)
    ax.set_ylabel('年份', fontsize=12)
    ax.set_title('广西月降水量热力图 (2005-2025)', fontsize=14, fontweight='bold')
    ax.set_xticklabels([f'{m}月' for m in range(1, 13)])
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/06_降水热力图.png')
    plt.close()

    # ==================== 1.8 四季对比 ====================
    print("  [1.8] 绘制四季气象要素对比图...")
    seasonal_avg = df.groupby('季节')[CORE_COLUMNS].mean().reindex(SEASON_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    metrics = [
        ('平均气温(°C)', '°C'), ('总降水量(mm)', 'mm'), ('平均湿度(%)', '%'),
        ('平均风速(km/h)', 'km/h'), ('日照时数(h)', 'h'), ('平均海平面气压(hPa)', 'hPa')
    ]
    for idx, (col, unit) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        vals = seasonal_avg[col].values
        bars = ax.bar(SEASON_ORDER, vals,
                      color=[SEASON_COLORS[s] for s in SEASON_ORDER], alpha=0.8, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(vals),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle('广西四季气象要素对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/07_季节性对比.png')
    plt.close()

    # ==================== 1.9 年际5年滑动平均 ====================
    print("  [1.9] 绘制年际5年滑动平均趋势图...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    rolling_cols = [
        ('平均气温(°C)', '#E74C3C'), ('总降水量(mm)', '#3498DB'),
        ('平均湿度(%)', '#2ECC71'), ('平均风速(km/h)', '#F39C12'),
        ('日照时数(h)', '#9B59B6'), ('参考蒸散量ET₀(mm)', '#1ABC9C')
    ]
    for idx, (col, color) in enumerate(rolling_cols):
        ax = axes[idx // 3, idx % 3]
        yearly_vals = df.groupby('年份_int')[col].mean()
        rolling_mean = yearly_vals.rolling(window=5, center=True).mean()
        ax.plot(yearly_vals.index, yearly_vals, 'o', color=color, alpha=0.3, markersize=4)
        ax.plot(rolling_mean.index, rolling_mean, '-', color=color, linewidth=2.5, label='5年滑动平均')
        z_val = np.polyfit(yearly_vals.dropna().index, yearly_vals.dropna().values, 1)
        p_line = np.poly1d(z_val)
        ax.plot(yearly_vals.index, p_line(yearly_vals.index), 'k--', linewidth=1.5, alpha=0.7,
                label=f'线性趋势 ({z_val[0]:+.4f}/年)')
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('广西气象要素年际变化趋势 (含5年滑动平均)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/08_年际滑动平均趋势.png')
    plt.close()

    # 保存结果
    results['desc_stats'] = desc
    results['yearly'] = yearly
    results['seasonal_avg'] = seasonal_avg
    print("  步骤1完成, 共生成8张图表。\n")
    return results
