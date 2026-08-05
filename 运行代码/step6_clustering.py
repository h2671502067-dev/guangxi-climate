"""
步骤6: 聚类分析
===============
对12个月份的气象特征进行聚类, 发现月份的气候分组模式。

数学方法:
  6.1 K-Means聚类:
    目标: 最小化簇内平方和 (WCSS)
      min Σ_{k=1}^{K} Σ_{xᵢ ∈ Cₖ} ||xᵢ - μₖ||²
    其中 μₖ 为簇k的质心, Cₖ 为簇k的样本集合。

    算法步骤:
      1. 随机初始化K个质心
      2. 分配: 将每个样本分配到最近质心的簇
      3. 更新: 重新计算每个簇的质心 μₖ = (1/|Cₖ|) Σ xᵢ
      4. 重复步骤2-3直到收敛

    最优K选择 (肘部法则 + 轮廓系数):
      肘部法则: 绘制K vs WCSS曲线, 对WCSS序列计算二阶差分,
        二阶差分绝对值最大处对应的K即为拐点。
        一阶差分: d₁(k) = WCSS(k) - WCSS(k-1)
        二阶差分: d₂(k) = d₁(k+1) - d₁(k)
        optimal_k = argmax_k |d₂(k)| + 2  (补偿两次差分偏移)

      轮廓系数: 衡量样本与其所属簇的紧密度及与其他簇的分离度。
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
        其中 a(i) 为样本i到同簇其他样本的平均距离,
        b(i) 为样本i到最近异簇所有样本的平均距离。
        取值范围 [-1, 1], 越大表示聚类效果越好。
        全局轮廓系数: S = (1/n) Σ s(i)

      注意: 本数据集仅含12个月份样本, 聚类结果仅供参考。

  6.2 PCA降维:
    目标: 找到正交投影方向, 使投影后方差最大化。
    协方差矩阵: Σ = (1/(n-1)) X'X
    特征值分解: Σ = V Λ V'
    主成分: Z = XV
    第k主成分解释方差比: λₖ / Σ λᵢ

  6.3 层次聚类 (Hierarchical Clustering):
    Ward方法: 合并使簇内方差增加最小的两个簇
    距离度量: 欧氏距离 d(x,y) = √[Σ(xᵢ - yᵢ)²]
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from config import (CLUSTER_FEATURES, CLUSTER_K_RANGE,
                    CHART_DIR, SEASON_COLORS, plt_config)

for k, v in plt_config.items():
    plt.rcParams[k] = v


def run(df, results):
    print("=" * 60)
    print("步骤6: 聚类分析")
    print("=" * 60)

    # ==================== 6.1 数据准备 ====================
    print("  [6.1] 准备月份聚类数据...")
    monthly_avg = df.groupby('月份_int')[CLUSTER_FEATURES].mean()
    scaler = StandardScaler()
    monthly_scaled = scaler.fit_transform(monthly_avg)
    n_samples = len(monthly_scaled)
    print(f"    样本量: n={n_samples} (仅{n_samples}个月份, 聚类结果仅供参考)")

    # ==================== 6.2 肘部法则 + 轮廓系数自动选K ====================
    print("  [6.2] 肘部法则与轮廓系数确定最优K值...")
    inertias = []
    silhouettes = []
    k_range_list = list(CLUSTER_K_RANGE)

    for k in CLUSTER_K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(monthly_scaled)
        inertias.append(km.inertia_)
        print(f"    K={k}: WCSS={km.inertia_:.2f}", end="")

        # 轮廓系数需要 k >= 2 且 k < n_samples
        if 2 <= k < n_samples:
            labels = km.fit_predict(monthly_scaled)
            sil = silhouette_score(monthly_scaled, labels)
            silhouettes.append(sil)
            print(f", 轮廓系数={sil:.4f}")
        else:
            silhouettes.append(-1)
            print(f", 轮廓系数=不可计算 (k需 < n_samples={n_samples})")

    # 肘部法则: 二阶差分最大处为拐点
    # 一阶差分: inertias的一阶差分
    first_diff = np.diff(inertias)
    # 二阶差分: 一阶差分的差分
    second_diff = np.diff(first_diff)
    # 二阶差分绝对值最大处索引, +2补偿两次diff的偏移
    optimal_k_idx = int(np.argmax(np.abs(second_diff))) + 2
    optimal_k = k_range_list[optimal_k_idx]

    # 验证: 检查最优K是否在有效范围内 (k < n_samples)
    if optimal_k >= n_samples:
        print(f"    警告: 肘部法则选出的K={optimal_k} >= 样本数{n_samples}, "
              f"回退至K={n_samples - 1}")
        optimal_k = n_samples - 1

    print(f"    >>> 自动选择最优K={optimal_k} (二阶差分拐点法)")
    if silhouettes[optimal_k_idx] >= 0:
        print(f"    >>> 对应轮廓系数={silhouettes[optimal_k_idx]:.4f}")

    # ==================== 6.3 K-Means聚类 ====================
    print(f"  [6.3] 执行K-Means聚类 (K={optimal_k})...")
    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    monthly_avg = monthly_avg.copy()
    monthly_avg['聚类'] = km_final.fit_predict(monthly_scaled)

    print(f"    聚类结果:")
    for cid in range(optimal_k):
        months = sorted(monthly_avg[monthly_avg['聚类'] == cid].index.tolist())
        print(f"      簇{cid}: {', '.join([f'{m}月' for m in months])}")

    # ==================== 6.4 PCA降维 ====================
    print("  [6.4] PCA降维分析...")
    pca = PCA(n_components=2)
    monthly_pca = pca.fit_transform(monthly_scaled)
    monthly_avg['PC1'] = monthly_pca[:, 0]
    monthly_avg['PC2'] = monthly_pca[:, 1]
    print(f"    PC1解释方差: {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"    PC2解释方差: {pca.explained_variance_ratio_[1]*100:.1f}%")
    print(f"    累计: {sum(pca.explained_variance_ratio_)*100:.1f}%")

    # ==================== 6.5 层次聚类 ====================
    print("  [6.5] 层次聚类 (Ward方法)...")
    linked = linkage(monthly_scaled, method='ward')

    # ==================== 6.6 可视化 ====================
    print("  [6.6] 绘制聚类分析图...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 子图1: 肘部法则 + 轮廓系数双轴图
    ax1 = axes[0, 0]
    color_wcss = '#E74C3C'
    color_sil = '#3498DB'

    ax1.plot(k_range_list, inertias, 'o-', color=color_wcss, linewidth=2,
             markersize=8, label='WCSS (簇内平方和)')
    ax1.axvline(optimal_k, color='gray', linestyle='--', alpha=0.6,
                label=f'最优K={optimal_k}')
    ax1.set_xlabel('聚类数 K')
    ax1.set_ylabel('WCSS', color=color_wcss)
    ax1.tick_params(axis='y', labelcolor=color_wcss)
    ax1.set_title('肘部法则与轮廓系数（样本量n=12，结果仅供参考）',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 轮廓系数共用x轴
    ax1b = ax1.twinx()
    valid_sil = [(k, s) for k, s in zip(k_range_list, silhouettes) if s >= 0]
    if valid_sil:
        sil_ks, sil_vals = zip(*valid_sil)
        ax1b.plot(sil_ks, sil_vals, 's--', color=color_sil, linewidth=2,
                  markersize=8, label='轮廓系数')
        ax1b.set_ylabel('轮廓系数', color=color_sil)
        ax1b.tick_params(axis='y', labelcolor=color_sil)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    # 子图2: K-Means聚类结果 (气温-降水)
    ax = axes[0, 1]
    colors_k = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12',
                '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']
    for cid in range(optimal_k):
        mask = monthly_avg['聚类'] == cid
        ax.scatter(monthly_avg.loc[mask, '平均气温(°C)'],
                   monthly_avg.loc[mask, '总降水量(mm)'],
                   c=colors_k[cid], s=120, label=f'簇{cid}',
                   edgecolors='white', linewidth=2)
        for m in monthly_avg[mask].index:
            ax.annotate(f'{m}月',
                        (monthly_avg.loc[m, '平均气温(°C)'],
                         monthly_avg.loc[m, '总降水量(mm)']),
                        fontsize=10, ha='center', va='bottom', fontweight='bold')
    ax.set_xlabel('平均气温 (°C)')
    ax.set_ylabel('平均降水量 (mm)')
    ax.set_title('K-Means聚类结果（样本量n=12，结果仅供参考）',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图3: PCA降维可视化
    ax = axes[1, 0]
    for cid in range(optimal_k):
        mask = monthly_avg['聚类'] == cid
        ax.scatter(monthly_avg.loc[mask, 'PC1'], monthly_avg.loc[mask, 'PC2'],
                   c=colors_k[cid], s=120, label=f'簇{cid}',
                   edgecolors='white', linewidth=2)
        for m in monthly_avg[mask].index:
            ax.annotate(f'{m}月',
                        (monthly_avg.loc[m, 'PC1'], monthly_avg.loc[m, 'PC2']),
                        fontsize=10, ha='center', va='bottom', fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA降维聚类可视化（样本量n=12，结果仅供参考）',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图4: 层次聚类树状图
    ax = axes[1, 1]
    dendrogram(linked, labels=[f'{m}月' for m in range(1, 13)], ax=ax,
               leaf_rotation=0, leaf_font_size=11)
    ax.set_title('层次聚类树状图（样本量n=12，结果仅供参考）',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Ward距离')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('广西月份气象特征聚类分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/17_聚类分析.png')
    plt.close()

    results['clustering'] = {
        'cluster_labels': monthly_avg['聚类'].to_dict(),
        'pca_variance': pca.explained_variance_ratio_.tolist(),
        'inertias': dict(zip(k_range_list, inertias)),
        'monthly_avg': monthly_avg,
        'optimal_k': optimal_k,
        'silhouettes': dict(zip(k_range_list, silhouettes)),
    }
    print("  步骤6完成, 共生成1张图表。\n")
    return results
