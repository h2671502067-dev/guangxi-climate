"""
步骤7: Mann-Kendall趋势检验
============================
对年际气象指标进行非参数趋势检验, 判断变化趋势的显著性。

数学方法:
  7.1 Mann-Kendall趋势检验:
    原假设 H₀: 无单调趋势
    备择假设 H₁: 存在单调趋势

    检验统计量 S:
      S = Σ_{i<j} sgn(xⱼ - xᵢ)
      其中 sgn(θ) = 1 (θ>0), 0 (θ=0), -1 (θ<0)

    方差 (含 ties 修正):
      Var(S) = [n(n-1)(2n+5) - Σtᵢ(tᵢ-1)(2tᵢ+5)] / 18
      其中 tᵢ 为第i组相同值的数量; 当数据无重复值时退化为经典公式。

    标准化检验统计量 Z:
      Z = (S-1)/√Var(S)  当 S > 0
      Z = 0               当 S = 0
      Z = (S+1)/√Var(S)  当 S < 0
      Z ~ N(0,1)

    判定:
      |Z| > Z_{α/2} (α=0.05时为1.96) → 拒绝H₀, 趋势显著
      Z > 0 → 上升趋势; Z < 0 → 下降趋势

  7.2 Sen斜率估计:
    Theil-Sen中位数斜率:
      β = median{ (xⱼ - xᵢ) / (j - i) }  对所有 i < j
    β > 0 → 上升趋势, β < 0 → 下降趋势
    Sen斜率对异常值具有鲁棒性。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy import stats
from config import TREND_COLUMNS, CHART_DIR, plt_config

for k, v in plt_config.items():
    plt.rcParams[k] = v


def mann_kendall_test(x):
    """执行Mann-Kendall趋势检验和Sen斜率估计"""
    n = len(x)
    x = np.array(x, dtype=float)

    # 计算S统计量
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(x[j] - x[i])

    # 方差 (含 ties 修正)
    value_counts = Counter(x)
    ties_correction = 0
    for count in value_counts.values():
        if count > 1:
            ties_correction += count * (count - 1) * (2 * count + 5)
    var_s = (n * (n - 1) * (2 * n + 5) - ties_correction) / 18

    # Sen斜率
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            slopes.append((x[j] - x[i]) / (j - i))
    sen_slope = np.median(slopes) if slopes else 0.0

    # 零值保护: 防止除零错误
    if var_s <= 0:
        return {'z': 0.0, 'p': 1.0, 'sen_slope': sen_slope, 's': s, 'var_s': var_s}

    # Z统计量
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    # p值 (双尾检验)
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return {'z': z, 'p': p, 'sen_slope': sen_slope, 's': s, 'var_s': var_s}


def run(df, results):
    print("=" * 60)
    print("步骤7: Mann-Kendall趋势检验")
    print("=" * 60)

    # ==================== 7.1 逐年均值计算 ====================
    print("  [7.1] 计算各指标年均值...")
    yearly_data = df.groupby('年份_int')[TREND_COLUMNS].mean()

    # ==================== 7.2 样本量警告 ====================
    n_years = len(yearly_data)
    print(f"\n  [警告] 当前样本量 n={n_years}, 统计功效偏低。")
    print(f"         Mann-Kendall检验在小样本(n<30)下可能无法有效")
    print(f"         检测中等强度的趋势, 结果仅供参考, 需结合其他")
    print(f"         方法综合判断。\n")

    # ==================== 7.3 执行MK检验 ====================
    print("  [7.3] 执行Mann-Kendall趋势检验...")
    trend_results = []
    print(f"\n  {'指标':<25} {'S统计量':>10} {'Z值':>8} {'p值':>10} {'Sen斜率':>12} {'趋势':>10}")
    print(f"  {'-'*80}")

    for col in TREND_COLUMNS:
        vals = yearly_data[col].dropna().values
        mk = mann_kendall_test(vals)

        if mk['p'] < 0.01:
            trend = '极显著' + ('上升' if mk['z'] > 0 else '下降')
        elif mk['p'] < 0.05:
            trend = '显著' + ('上升' if mk['z'] > 0 else '下降')
        elif mk['p'] < 0.1:
            trend = '边缘' + ('上升' if mk['z'] > 0 else '下降')
        else:
            trend = '不显著'

        trend_results.append({
            'column': col, **mk, 'trend': trend,
            'yearly_vals': vals, 'years': yearly_data[col].dropna().index.tolist()
        })

        print(f"  {col:<25} {mk['s']:>10.0f} {mk['z']:>8.3f} {mk['p']:>10.4f} "
              f"{mk['sen_slope']:>12.4f} {trend:>10}")

    # ==================== 7.4 可视化 ====================
    print("  [7.4] 绘制趋势检验结果图...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes_flat = axes.flatten()

    for idx, tr in enumerate(trend_results):
        ax = axes_flat[idx]
        years = tr['years']
        vals = tr['yearly_vals']

        # 原始数据
        ax.plot(years, vals, 'o-', color='#333', markersize=4, linewidth=0.8, label='年均值')

        # Sen斜率趋势线（标准截距计算：median(y_i - slope * x_i)）
        sen_intercept = np.median(np.array(vals) - tr['sen_slope'] * np.array(years))
        trend_line = sen_intercept + tr['sen_slope'] * np.array(years)
        color = '#E74C3C' if tr['sen_slope'] > 0 else '#3498DB'
        ax.plot(years, trend_line, '--', color=color, linewidth=2,
                label=f"Sen斜率={tr['sen_slope']:+.4f}/年")

        # 标注显著性
        sig_text = f"Z={tr['z']:.2f}\np={tr['p']:.4f}\n{tr['trend']}"
        ax.text(0.02, 0.98, sig_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_title(tr['column'], fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    if len(trend_results) < len(axes_flat):
        for idx in range(len(trend_results), len(axes_flat)):
            axes_flat[idx].set_visible(False)

    fig.suptitle('广西气象指标Mann-Kendall趋势检验 (2005-2025)\n'
                 f'[注] 样本量 n={n_years}, 统计功效偏低, 结果仅供参考',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/18_MannKendall趋势检验.png')
    plt.close()

    results['trend'] = trend_results
    print("  步骤7完成, 共生成1张图表。\n")
    return results
