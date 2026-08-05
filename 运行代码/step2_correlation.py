"""
步骤2: 相关性分析
=================
分析核心气象要素之间的线性相关关系。

数学方法:
  Pearson相关系数:
    r_{xy} = Σ(xᵢ - x̄)(yᵢ - ȳ) / √[Σ(xᵢ - x̄)² × Σ(yᵢ - ȳ)²]
    r ∈ [-1, 1], |r| > 0.8 强相关, 0.5 < |r| < 0.8 中等相关, |r| < 0.5 弱相关

  显著性检验:
    t = r × √(n-2) / √(1-r²), 服从 t(n-2) 分布
    p < 0.05 时相关性显著
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from config import CORE_COLUMNS, CHART_DIR, plt_config

for k, v in plt_config.items():
    plt.rcParams[k] = v


def run(df, results):
    print("=" * 60)
    print("步骤2: 相关性分析")
    print("=" * 60)

    # ==================== 2.1 Pearson相关系数矩阵 ====================
    print("  [2.1] 计算Pearson相关系数矩阵...")
    corr_matrix = df[CORE_COLUMNS].corr()
    cols = corr_matrix.columns.tolist()

    # 提取强相关对
    strong_pos, strong_neg = [], []
    # [修复#14] 计算p值矩阵
    p_matrix = pd.DataFrame(np.zeros_like(corr_matrix), index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            valid = df[[cols[i], cols[j]]].dropna()
            if len(valid) < 3:
                continue
            r, p_val = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
            p_matrix.iloc[i, j] = p_val
            p_matrix.iloc[j, i] = p_val
            if r > 0.8:
                strong_pos.append((cols[i], cols[j], r, p_val))
            elif r < -0.7:
                strong_neg.append((cols[i], cols[j], r, p_val))

    print(f"  强正相关对 (r > 0.8): {len(strong_pos)} 对")
    for a, b, r, p in sorted(strong_pos, key=lambda x: -x[2])[:5]:
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        print(f"    {a} ↔ {b}: r = {r:.4f} (p={p:.2e} {sig})")
    print(f"  强负相关对 (r < -0.7): {len(strong_neg)} 对")
    for a, b, r, p in sorted(strong_neg, key=lambda x: x[2])[:5]:
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        print(f"    {a} ↔ {b}: r = {r:.4f} (p={p:.2e} {sig})")

    # ==================== 2.2 相关性热力图 ====================
    print("  [2.2] 绘制相关性热力图...")
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Pearson相关系数', 'shrink': 0.8})
    ax.set_title('广西气象要素Pearson相关性矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/09_相关性矩阵.png')
    plt.close()

    # ==================== 2.3 关键变量对散点图 ====================
    print("  [2.3] 绘制关键变量对散点图...")
    scatter_pairs = [
        ('平均气温(°C)', '土壤温度0-7cm(°C)', '气温与土壤温度', '#E74C3C'),
        ('平均气温(°C)', '平均海平面气压(hPa)', '气温与气压', '#3498DB'),
        ('总降水量(mm)', '平均湿度(%)', '降水与湿度', '#2ECC71'),
        ('日照时数(h)', '平均云量(%)', '日照与云量', '#F39C12'),
        ('平均气温(°C)', '参考蒸散量ET₀(mm)', '气温与蒸散量', '#9B59B6'),
        ('平均风速(km/h)', '最大阵风(km/h)', '平均风速与阵风', '#1ABC9C'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plot_idx = 0
    for idx, (x_col, y_col, title, color) in enumerate(scatter_pairs):
        # 检查列是否存在
        if x_col not in df.columns or y_col not in df.columns:
            continue
        ax = axes[plot_idx // 3, plot_idx % 3]
        ax.scatter(df[x_col], df[y_col], alpha=0.5, s=20, c=color)
        r = np.corrcoef(df[x_col], df[y_col])[0, 1]
        z = np.polyfit(df[x_col], df[y_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=1.5, label=f'r = {r:.3f}')
        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    fig.suptitle('广西关键气象变量对散点图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/10_关键变量散点图.png')
    plt.close()

    results['corr_matrix'] = corr_matrix
    results['strong_pos'] = strong_pos
    results['strong_neg'] = strong_neg
    print("  步骤2完成, 共生成2张图表。\n")
    return results
