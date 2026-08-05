"""
步骤3: 时间序列分解与平稳性检验
=================================
将时间序列分解为趋势、季节和残差分量, 并进行ADF平稳性检验。

数学方法:
  3.1 时间序列分解 (加法模型):
    X(t) = T(t) + S(t) + R(t)
    其中 T(t) 为趋势分量, S(t) 为季节分量 (周期=12), R(t) 为残差分量
    采用移动平均方法提取趋势, 季节性分解使用 STL/经典分解法。

  3.2 ADF检验 (Augmented Dickey-Fuller):
    原假设 H₀: 序列存在单位根 (非平稳)
    备择假设 H₁: 序列不存在单位根 (平稳)
    检验回归: Δyₜ = α + βt + γyₜ₋₁ + Σ δᵢ Δyₜ₋ᵢ + εₜ
    若 ADF统计量 < 临界值 (或 p < 0.05), 则拒绝H₀, 序列平稳。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from config import CHART_DIR, plt_config

for k, v in plt_config.items():
    plt.rcParams[k] = v


def run(df, results):
    print("=" * 60)
    print("步骤3: 时间序列分解与平稳性检验")
    print("=" * 60)

    # ==================== 3.1 气温时间序列分解 ====================
    print("  [3.1] 气温时间序列分解 (加法模型, 周期=12)...")
    ts_temp = df.set_index('日期')['平均气温(°C)']
    # [修复#15] 删除弃用的freq参数

    decomp_temp = seasonal_decompose(ts_temp, model='additive', period=12)

    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    decomp_temp.observed.plot(ax=axes[0], color='#333')
    axes[0].set_title('原始序列', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    decomp_temp.trend.plot(ax=axes[1], color='#E74C3C')
    axes[1].set_title('趋势分量 T(t)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    decomp_temp.seasonal.plot(ax=axes[2], color='#3498DB')
    axes[2].set_title('季节分量 S(t)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    decomp_temp.resid.plot(ax=axes[3], color='#2ECC71')
    axes[3].set_title('残差分量 R(t)', fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    fig.suptitle('广西月平均气温时间序列分解: X(t) = T(t) + S(t) + R(t)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/11_气温时间序列分解.png')
    plt.close()

    # ==================== 3.2 气温ADF检验 ====================
    print("  [3.2] 气温ADF平稳性检验...")
    adf_temp = adfuller(ts_temp.dropna())
    print(f"    ADF统计量: {adf_temp[0]:.4f}")
    print(f"    p值: {adf_temp[1]:.6f}")
    print(f"    滞后阶数: {adf_temp[2]}")
    print(f"    结论: {'序列平稳 (拒绝H₀)' if adf_temp[1] < 0.05 else '序列非平稳 (不能拒绝H₀)'}")

    # ==================== 3.3 降水时间序列分解 ====================
    print("  [3.3] 降水时间序列分解 (加法模型, 周期=12)...")
    ts_precip = df.set_index('日期')['总降水量(mm)']
    # [修复#15] 删除弃用的freq参数

    decomp_precip = seasonal_decompose(ts_precip, model='additive', period=12)

    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    decomp_precip.observed.plot(ax=axes[0], color='#333')
    axes[0].set_title('原始序列', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    decomp_precip.trend.plot(ax=axes[1], color='#3498DB')
    axes[1].set_title('趋势分量 T(t)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    decomp_precip.seasonal.plot(ax=axes[2], color='#3498DB')
    axes[2].set_title('季节分量 S(t)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    decomp_precip.resid.plot(ax=axes[3], color='#2ECC71')
    axes[3].set_title('残差分量 R(t)', fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    fig.suptitle('广西月降水量时间序列分解: X(t) = T(t) + S(t) + R(t)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/12_降水时间序列分解.png')
    plt.close()

    # ==================== 3.4 降水ADF检验 ====================
    print("  [3.4] 降水ADF平稳性检验...")
    adf_precip = adfuller(ts_precip.dropna())
    print(f"    ADF统计量: {adf_precip[0]:.4f}")
    print(f"    p值: {adf_precip[1]:.6f}")
    print(f"    结论: {'序列平稳 (拒绝H₀)' if adf_precip[1] < 0.05 else '序列非平稳 (不能拒绝H₀)'}")

    results['adf_temp'] = adf_temp
    results['adf_precip'] = adf_precip
    results['decomp_temp'] = decomp_temp
    results['decomp_precip'] = decomp_precip
    print("  步骤3完成, 共生成2张图表。\n")
    return results
