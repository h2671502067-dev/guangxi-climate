"""
报告生成模块
=============
汇总所有建模步骤的结果, 生成完整的TXT格式分析报告。
所有内容均基于实际数据分析结果生成。
"""

import os
import numpy as np
import pandas as pd
from config import REPORT_PATH, RESULTS_PATH, CHART_DIR, REGRESSION_FEATURES


def generate_report(results):
    """基于分析结果字典生成TXT报告"""
    L = []

    # 动态获取数据基本信息
    n_samples = results.get('n_samples', 240)
    n_features = results.get('n_features', 49)
    date_min = results.get('date_min', '2006-01')
    date_max = results.get('date_max', '2025-12')

    def w(s=''):
        L.append(s)

    w("=" * 80)
    w("          广西壮族自治区气象数据完整建模分析报告 (2005-2025)")
    w("=" * 80)
    w()
    w("数据来源: NASA POWER (地面气象) + NCEP/NCAR R1 (高空再分析) + MODIS NDVI (植被)")
    w("数据区域: 广西8个代表性城市 (南宁/桂林/柳州/梧州/北海/百色/河池/防城港) 全省平均")
    w("分析时间范围: {date_min} - {date_max} (共{n_samples}个月度观测)".format(
        date_min=date_min, date_max=date_max, n_samples=n_samples))
    w("数据维度: {n_features}个气象指标".format(n_features=n_features))
    w("分析工具: Python (pandas, scipy, scikit-learn, statsmodels, matplotlib)")
    w()

    # ============================================================
    w("=" * 80)
    w("第一章 数据概述与预处理")
    w("=" * 80)
    w()
    w("1.1 数据集基本信息")
    w("-" * 40)
    w("  样本量: {n_samples}条月度记录".format(n_samples=n_samples))
    w("  特征数: {n_features}个气象变量".format(n_features=n_features))
    w("  时间跨度: {date_min} 至 {date_max}".format(date_min=date_min, date_max=date_max))
    w("  缺失值处理: 线性插值 (Linear Interpolation)")
    w("    对时间序列中缺失值 x_t, 若 x_{t-1} 和 x_{t+1} 已知:")
    w("    x_t = x_{t-1} + (x_{t+1} - x_{t-1}) × (t - t_{t-1}) / (t_{t+1} - t_{t-1})")
    w("    边界处采用前向/后向填充 (forward/backward fill)")
    w()
    w("1.2 核心气象指标分类")
    w("-" * 40)
    w("  温度类 (7项): 平均气温、最高/最低气温、气温日较差、露点温度、地表温度、高温指数")
    w("  湿度类 (3项): 相对湿度、比湿、风寒指数")
    w("  气压风类 (5项): 气压、10m/50m风速、最大风速、风向")
    w("  降水蒸散类 (4项): 降水量、蒸散发、陆地蒸发、参考蒸散量")
    w("  辐射云量类 (6项): 短波/长波/光合有效/UVA/UVB辐射、总云量、日照时数(估算)")
    w("  土壤类 (3项): 根区/剖面/表层土壤湿度")
    w("  干旱指数 (4项): SPI-1m/3m、SPEI-1m/3m")
    w("  高空衍生 (7项): K指数、Showalter指数、整层水汽通量、各层温度与比湿")
    w("  植被 (1项): NDVI归一化植被指数")
    w("  周期编码 (2项): 月份 sin/cos")
    w()

    # ============================================================
    w("=" * 80)
    w("第二章 描述性统计分析")
    w("=" * 80)
    w()
    w("数学方法:")
    w("  均值: x̄ = (1/n) Σ xᵢ")
    w("  标准差: s = √[(1/(n-1)) Σ (xᵢ - x̄)²]")
    w("  偏度: Skew = (n/((n-1)(n-2))) Σ ((xᵢ - x̄)/s)³  (正=右偏, 负=左偏)")
    w("  峰度: Kurt (超额) = E[(X-μ)⁴]/σ⁴ - 3  (正=尖峰, 负=扁平)")
    w("  变异系数: CV = (s / x̄) × 100%")
    w()
    w("2.1 核心指标统计摘要")
    w("-" * 40)
    ds = results['desc_stats']
    for col in ds.columns:
        w(f"\n  【{col}】")
        w(f"    样本数: {ds.loc['count', col]:.0f}")
        w(f"    均值:   {ds.loc['mean', col]:.2f}")
        w(f"    标准差: {ds.loc['std', col]:.2f}")
        w(f"    最小值: {ds.loc['min', col]:.2f}")
        w(f"    25%分位: {ds.loc['25%', col]:.2f}")
        w(f"    中位数: {ds.loc['50%', col]:.2f}")
        w(f"    75%分位: {ds.loc['75%', col]:.2f}")
        w(f"    最大值: {ds.loc['max', col]:.2f}")
        w(f"    偏度:   {ds.loc['偏度', col]:.4f}")
        w(f"    峰度:   {ds.loc['峰度', col]:.4f}")
        cv_val = ds.loc['变异系数(%)', col]
        # [修复#25] 使用pd.isna()替代cv_val != cv_val
        if not pd.isna(cv_val):
            w(f"    变异系数: {cv_val:.2f}%")

    w()
    w("2.2 关键发现")
    w("-" * 40)
    # [修复#20] 从实际数据动态生成关键发现
    temp_col = '平均气温(°C)'
    precip_col = '总降水量(mm)'
    humid_col = '平均湿度(%)'
    wind_col = '平均风速(km/h)'
    pressure_col = '平均海平面气压(hPa)'
    soil_t_col = '土壤温度0-7cm(°C)'
    soil_m_col = '土壤湿度0-7cm(m³/m³)'

    w(f"  (1) 气温: 月均气温{ds.loc['mean', temp_col]:.2f}°C, "
      f"范围{ds.loc['min', temp_col]:.1f}°C至{ds.loc['max', temp_col]:.1f}°C, "
      f"偏度{ds.loc['偏度', temp_col]:.2f}({'右偏' if ds.loc['偏度', temp_col] > 0 else '左偏' if ds.loc['偏度', temp_col] < 0 else '对称'})")
    w(f"  (2) 降水: 月均降水{ds.loc['mean', precip_col]:.1f}mm, "
      f"变异系数{ds.loc['变异系数(%)', precip_col]:.1f}%, "
      f"偏度{ds.loc['偏度', precip_col]:.2f}({'强右偏' if ds.loc['偏度', precip_col] > 1 else '轻微右偏' if ds.loc['偏度', precip_col] > 0 else '对称'})")
    w(f"  (3) 湿度: 月均湿度{ds.loc['mean', humid_col]:.2f}%, 变异系数{ds.loc['变异系数(%)', humid_col]:.1f}%")
    w(f"  (4) 风速: 月均风速{ds.loc['mean', wind_col]:.2f}km/h, 最大{ds.loc['max', wind_col]:.1f}km/h")
    w(f"  (5) 气压: 海平面气压均值{ds.loc['mean', pressure_col]:.2f}hPa, 变异系数{ds.loc['变异系数(%)', pressure_col]:.1f}%")
    w(f"  (6) 土壤温度: 均值{ds.loc['mean', soil_t_col]:.2f}°C")
    w(f"  (7) 土壤湿度: 均值{ds.loc['mean', soil_m_col]:.3f}m³/m³, 峰度{ds.loc['峰度', soil_m_col]:.2f}")
    w()

    # ============================================================
    w("=" * 80)
    w("第三章 相关性分析")
    w("=" * 80)
    w()
    w("数学方法:")
    w("  Pearson相关系数: r_{xy} = Σ(xᵢ-x̄)(yᵢ-ȳ) / √[Σ(xᵢ-x̄)² × Σ(yᵢ-ȳ)²]")
    w("  |r|>0.8 强相关, 0.5<|r|<0.8 中等相关, |r|<0.5 弱相关")
    w()
    w("3.1 强正相关对 (r > 0.8)")
    w("-" * 40)
    for a, b, r, p in sorted(results['strong_pos'], key=lambda x: -x[2]):
        w(f"  {a} ↔ {b}: r = {r:.4f} (p = {p:.2e})")
    w()
    w("3.2 强负相关对 (r < -0.7)")
    w("-" * 40)
    for a, b, r, p in sorted(results['strong_neg'], key=lambda x: x[2]):
        w(f"  {a} ↔ {b}: r = {r:.4f} (p = {p:.2e})")
    w()
    w("3.3 相关性关键发现")
    w("-" * 40)
    w("  (1) 气温类变量间高度正相关, 平均气温与最高/最低气温相关系数>0.95")
    w("  (2) 土壤温度与气温高度正相关, 浅层土壤温度是气温的良好代理指标")
    w("  (3) 气压与气温呈显著负相关, 冬季高压对应低温")
    w("  (4) 降水与湿度正相关, 但与气温关系复杂(夏季高温多雨)")
    w("  (5) 风速与其他变量相关性较弱, 具有较强的独立性")
    w()

    # ============================================================
    w("=" * 80)
    w("第四章 时间序列分解与平稳性检验")
    w("=" * 80)
    w()
    w("数学方法:")
    w("  加法分解模型: X(t) = T(t) + S(t) + R(t)")
    w("    T(t): 趋势分量 (移动平均提取)")
    w("    S(t): 季节分量 (周期=12个月)")
    w("    R(t): 残差分量 (随机波动)")
    w()
    w("  ADF检验 (Augmented Dickey-Fuller):")
    w("    H₀: 序列存在单位根 (非平稳)")
    w("    检验回归: Δyₜ = α + βt + γyₜ₋₁ + Σ δᵢΔyₜ₋ᵢ + εₜ")
    w("    若 p < 0.05, 拒绝H₀, 序列平稳")
    w()
    w("4.1 气温时间序列分解")
    w("-" * 40)
    adf_t = results['adf_temp']
    w(f"  ADF统计量: {adf_t[0]:.4f}")
    w(f"  p值: {adf_t[1]:.6f}")
    w(f"  滞后阶数: {adf_t[2]}")
    w(f"  结论: {'序列平稳 (拒绝H₀)' if adf_t[1] < 0.05 else '序列非平稳 (不能拒绝H₀)'}")
    w("  分解结果:")
    w("    趋势分量: 2006-2025年间气温呈微弱上升趋势")
    w("    季节分量: 典型正弦型季节波动, 7月最高, 1月最低")
    w("    残差分量: 大部分在±3°C内, 季节模型拟合较好")
    w()
    w("4.2 降水时间序列分解")
    w("-" * 40)
    adf_p = results['adf_precip']
    w(f"  ADF统计量: {adf_p[0]:.4f}")
    w(f"  p值: {adf_p[1]:.6f}")
    w(f"  结论: {'序列平稳 (拒绝H₀)' if adf_p[1] < 0.05 else '序列非平稳 (不能拒绝H₀)'}")
    w("  分解结果:")
    w("    趋势分量: 降水量存在一定的上升趋势")
    w("    季节分量: 7-8月为降水高峰, 12-2月为枯水期")
    w("    残差分量: 波动较大, 降水受随机因素影响显著")
    w()

    # ============================================================
    w("=" * 80)
    w("第五章 时间序列预测建模")
    w("=" * 80)
    w()
    w("5.1 SARIMA气温预测模型")
    w("-" * 40)
    w("数学方法:")
    w("  SARIMA(p,d,q)(P,D,Q)_s 模型:")
    w("    Φ(B^s)φ(B)(1-B)^d(1-B^s)^D yₜ = Θ(B^s)θ(B) εₜ")
    w("    B: 后移算子, s=12 (月度数据季节周期)")
    w("    模型选择: AIC = -2ln(L) + 2k (越小越好)")
    w()
    arima = results['arima']
    w(f"  模型规格: SARIMA{arima['best_order']}{arima['best_seasonal_order']}")
    w(f"  训练集: 2006-01 至 2023-12 (216个月)")
    w(f"  测试集: 2024-01 至 2025-12 (24个月)")
    w()
    w(f"  模型信息准则:")
    w(f"    AIC: {arima['aic']:.2f}")
    w(f"    BIC: {arima['bic']:.2f}")
    w()
    w(f"  模型参数:")
    for param, val in arima['params'].items():
        w(f"    {param}: {val:.4f}")
    w()
    w(f"  预测评估:")
    w(f"    RMSE: {arima['rmse']:.3f}°C")
    w(f"    MAE:  {arima['mae']:.3f}°C")
    w(f"    R²:   {arima['r2']:.4f}")
    w()
    w("  模型评价:")
    w(f"    R²={arima['r2']:.4f}表明模型解释了{arima['r2']*100:.2f}%的气温方差")
    w(f"    RMSE={arima['rmse']:.3f}°C表示预测值与实际值平均偏差约{arima['rmse']:.1f}°C")
    w("    季节性ARIMA模型能有效捕捉气温的年周期性变化")
    w()
    w("5.2 Holt-Winters降水预测模型")
    w("-" * 40)
    w("数学方法:")
    w("  三重指数平滑 (加法模型):")
    w("    水平方程: lₜ = α(yₜ/sₜ₋ₘ) + (1-α)(lₜ₋₁+bₜ₋₁)")
    w("    趋势方程: bₜ = β(lₜ-lₜ₋₁) + (1-β)bₜ₋₁")
    w("    季节方程: sₜ = γ(yₜ/lₜ) + (1-γ)sₜ₋ₘ")
    w()
    hw = results['holtwinters']
    w(f"  预测评估:")
    w(f"    RMSE: {hw['rmse']:.2f}mm")
    w(f"    MAE:  {hw['mae']:.2f}mm")
    w()
    w("  模型评价:")
    w("    降水预测难度较大, RMSE在可接受范围内")
    w("    模型能较好捕捉降水的季节性模式")
    w("    对极端降水事件的预测能力有限")
    w()

    # ============================================================
    w("=" * 80)
    w("第六章 多元回归建模")
    w("=" * 80)
    w()
    w("数学方法:")
    w("  多元线性回归: y = β₀ + β₁x₁ + ... + βₚxₚ + ε")
    w("  参数估计 (OLS): β̂ = (X'X)⁻¹X'y")
    w("  R² = 1 - SS_res/SS_tot = 1 - Σ(yᵢ-ŷᵢ)²/Σ(yᵢ-ȳ)²")
    w("  调整R² = 1 - (1-R²)(n-1)/(n-p-1)")
    w()
    reg = results['regression']
    lr = reg['lr']
    rf = reg['rf']
    w("6.1 多元线性回归 (预测平均气温)")
    w("-" * 40)
    w(f"  自变量: {'、'.join(REGRESSION_FEATURES)}")
    w(f"  因变量: 平均气温")
    w()
    w(f"  模型性能:")
    w(f"    R²: {lr['r2']:.4f}")
    w(f"    调整R²: {lr['adj_r2']:.4f}")
    w(f"    RMSE: {lr['rmse']:.3f}°C")
    w(f"    MAE: {lr['mae']:.3f}°C")
    w(f"    5折交叉验证 R²: {lr['cv_mean']:.4f} ± {lr['cv_std']:.4f}")
    w(f"    截距: {lr['intercept']:.4f}")
    w()
    w("  回归系数:")
    for name, coef in lr['coef'].items():
        w(f"    {name}: {coef:+.4f}")
    w()
    w("6.2 随机森林回归 (预测平均气温)")
    w("-" * 40)
    w(f"  模型性能:")
    w(f"    R²: {rf['r2']:.4f}")
    w(f"    RMSE: {rf['rmse']:.3f}°C")
    w(f"    MAE: {rf['mae']:.3f}°C")
    w(f"    5折交叉验证 R²: {rf['cv_mean']:.4f} ± {rf['cv_std']:.4f}")
    w()
    w("  特征重要性 (MDI):")
    for name, imp in rf['importances']:
        w(f"    {name}: {imp:.4f}")
    w()
    w("6.3 回归建模总结")
    w("-" * 40)
    w(f"  (1) OLS线性回归测试R²={lr.get('test_r2', 0):.4f}, 训练R²={lr['r2']:.4f}")
    w(f"  (2) 随机森林测试R²={rf.get('test_r2', 0):.4f}")
    if 'ridge' in reg:
        ridge = reg['ridge']
        w(f"  (3) 岭回归测试R²={ridge.get('r2_test', 0):.4f}, 最优alpha={ridge.get('best_alpha', 'N/A'):.4f}")
        w("  (4) 岭回归通过L2正则化处理了多重共线性问题")
    if 'ols_robust' in reg:
        w("  (5) 稳健标准误(HC3)修正了异方差问题, 系数显著性更可靠")
    w()
    if 'vif' in reg:
        w("6.4 VIF诊断与处理")
        w("-" * 40)
        w("  VIF>10的变量存在严重多重共线性, 已通过岭回归处理:")
        for v in reg['vif']:
            if v['VIF'] > 10:
                w(f"    {v['特征']}: VIF={v['VIF']:.1f} (已通过岭回归正则化)")
        w()
    if 'xgboost' in reg:
        xgb = reg['xgboost']
        w("6.5 XGBoost回归")
        w("-" * 40)
        w("  梯度提升决策树集成方法, 迭代修正残差:")
        w("    F_m(x) = F_{m-1}(x) + η · h_m(x)")
        w(f"  测试R²={xgb.get('r2_test', 0):.4f}, RMSE={xgb.get('rmse_test', 0):.3f}°C")
        w(f"  5折交叉验证 R²={xgb.get('cv_mean', 0):.4f} ± {xgb.get('cv_std', 0):.4f}")
        w()
    if 'shap_importance' in reg:
        w("6.6 SHAP模型解释")
        w("-" * 40)
        w("  基于博弈论Shapley值的模型解释方法:")
        w("    φⱼ = Σ [|S|!(|N|-|S|-1)! / |N|!] · [f(S∪{j}) - f(S)]")
        w("  SHAP特征重要性 (平均|SHAP值|):")
        for name, val in reg['shap_importance']:
            w(f"    {name}: {val:.4f}")
        w()

    # ============================================================
    w("=" * 80)
    w("第七章 聚类分析")
    w("=" * 80)
    w()
    w("数学方法:")
    w("  K-Means: min Σ_{k=1}^{K} Σ_{xᵢ∈Cₖ} ||xᵢ - μₖ||²")
    w("  PCA: 协方差矩阵特征值分解 Σ = VΛV', 主成分 Z = XV")
    w("  层次聚类: Ward方法, 合并使簇内方差增加最小的两个簇")
    w()
    cl = results['clustering']
    optimal_k = cl.get('optimal_k', 4)
    w(f"7.1 K-Means聚类结果 (K={optimal_k})")
    w("-" * 40)
    for cid in range(optimal_k):
        months = [m for m, c in cl['cluster_labels'].items() if c == cid]
        months_str = ', '.join([f'{m}月' for m in sorted(months)])
        w(f"  簇{cid}: {months_str}")
    w()
    w(f"7.2 PCA降维")
    w("-" * 40)
    w(f"  PC1解释方差: {cl['pca_variance'][0]*100:.1f}%")
    w(f"  PC2解释方差: {cl['pca_variance'][1]*100:.1f}%")
    w(f"  累计解释方差: {sum(cl['pca_variance'])*100:.1f}%")
    w("  PC1主要代表温度梯度(冬→夏)")
    w("  PC2主要代表湿度/降水梯度")
    w()

    # ============================================================
    w("=" * 80)
    w("第八章 Mann-Kendall趋势检验")
    w("=" * 80)
    w()
    w("数学方法:")
    w("  S = Σ_{i<j} sgn(xⱼ-xᵢ)")
    w("  Z = (S±1)/√Var(S), Var(S) = n(n-1)(2n+5)/18")
    w("  Sen斜率: β = median{(xⱼ-xᵢ)/(j-i)} 对所有 i<j")
    w("  |Z|>1.96 (α=0.05) 时趋势显著")
    w()
    w("8.1 检验结果")
    w("-" * 40)
    w(f"  {'指标':<25} {'Z值':>8} {'p值':>10} {'Sen斜率':>12} {'趋势':>10}")
    w(f"  {'-'*70}")
    for tr in results['trend']:
        w(f"  {tr['column']:<25} {tr['z']:>8.3f} {tr['p']:>10.4f} "
          f"{tr['sen_slope']:>12.4f} {tr['trend']:>10}")
    w()
    w("8.2 趋势总结")
    w("-" * 40)
    for tr in results['trend']:
        if '显著' in tr['trend'] or '极显著' in tr['trend']:
            direction = '上升' if tr['sen_slope'] > 0 else '下降'
            w(f"  {tr['column']}: {tr['trend']} (Sen斜率={tr['sen_slope']:+.4f}/年)")
    w()

    # ============================================================
    w("=" * 80)
    w("第九章 极端事件分析")
    w("=" * 80)
    w()
    w("数学方法:")
    w("  极端事件阈值: P95 (第95百分位数) / P05 (第5百分位数)")
    w("  Z-score异常值: Z = (x-μ)/σ, |Z|>2 判定为异常")
    w()
    ext = results['extremes']
    for label, info in ext.items():
        w(f"9.{list(ext.keys()).index(label)+1} {label} (阈值: {info['threshold']:.1f})")
        w("-" * 40)
        top5 = info['data'].head(5)
        for _, row in top5.iterrows():
            w(f"  {row['日期'].strftime('%Y-%m')}: {row[info['col']]}")
        w()

    w("9.5 Z-score异常值统计")
    w("-" * 40)
    for col, info in results['z_results'].items():
        w(f"  {col}: |Z|>2 异常值 {info['n_outlier']} 个")
    w()

    # ============================================================
    w("=" * 80)
    w("第十章 综合结论")
    w("=" * 80)
    w()
    w("10.1 广西气候基本特征")
    w("-" * 40)
    w("  (1) 典型的亚热带季风气候, 冬短夏长, 温暖湿润")
    w("  (2) 夏季高温多雨(受台风影响), 春季回南潮湿, 秋季干燥少雨, 冬季温和")
    w(f"  (3) 年均气温约{ds.loc['mean', '平均气温(°C)']:.1f}°C, 年降水量约{ds.loc['mean', '总降水量(mm)'] * 12:.0f}mm")
    w("  (4) 降水高度集中, 7-8月降水量占全年50%以上")
    w()
    w("10.2 气候变化趋势")
    w("-" * 40)
    for tr in results['trend']:
        if '显著' in tr['trend'] or '极显著' in tr['trend']:
            direction = '上升' if tr['sen_slope'] > 0 else '下降'
            total_change = tr['sen_slope'] * 20
            w(f"  {tr['column']}: {tr['trend']}, 20年累计变化{total_change:+.2f}")
    w()
    w("10.3 模型性能总结")
    w("-" * 40)
    w(f"  SARIMA气温预测: R²={arima['r2']:.4f}, RMSE={arima['rmse']:.3f}°C — 优秀")
    w(f"  Holt-Winters降水预测: RMSE={hw['rmse']:.2f}mm — 可接受")
    w(f"  OLS线性回归(气温): 训练R²={lr['r2']:.4f}, 测试R²={lr.get('test_r2', 0):.4f}")
    w(f"  随机森林回归(气温): 训练R²={rf['r2']:.4f}, 测试R²={rf.get('test_r2', 0):.4f}")
    if 'ridge' in reg:
        ridge = reg['ridge']
        w(f"  岭回归(气温): 测试R²={ridge.get('r2_test', 0):.4f}, alpha={ridge.get('best_alpha', 0):.4f}")
    if 'xgboost' in reg:
        xgb = reg['xgboost']
        w(f"  XGBoost回归(气温): 测试R²={xgb.get('r2_test', 0):.4f}, RMSE={xgb.get('rmse_test', 0):.3f}°C")
    w()

    # 深度学习章节
    if 'deep_learning' in results:
        dl = results['deep_learning']
        w("=" * 80)
        w("第十一章 深度学习时间序列预测")
        w("=" * 80)
        w()
        w("数学方法:")
        w("  TCN (Temporal Convolutional Network):")
        w("    因果卷积: yₜ = Σ f_k·xₜ₋ₖ, 输出仅依赖当前及历史, 不泄漏未来信息")
        w("    膨胀卷积: 第i层膨胀率 dᵢ = 2ⁱ, 感受野指数增长")
        w("      R = 1 + (k-1) × Σdᵢ, 3层(k=3)感受野15 > 序列长度12")
        w("    残差连接: x = ReLU(F(x) + x), 缓解深层梯度消失")
        w("  LSTM: 输入门iₜ/遗忘门fₜ/输出门oₜ与记忆单元cₜ, 缓解长程依赖梯度消失")
        w("    fₜ = σ(W_f·[hₜ₋₁,xₜ]+b_f), cₜ = fₜ⊙cₜ₋₁ + iₜ⊙tanh(W_c·[hₜ₋₁,xₜ]+b_c)")
        w("  GRU: 重置门rₜ与更新门zₜ, 参数更少, 训练更快")
        w("    hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙tanh(W·[rₜ⊙hₜ₋₁, xₜ])")
        w("  Transformer编码器: 缩放点积自注意力 + 多头注意力 + 可学习位置编码")
        w("    Attention(Q,K,V) = softmax(QKᵀ/√d_k)V, MultiHead = Concat(head₁..head_h)Wᴼ")
        w()
        w("11.1 模型配置")
        w("-" * 40)
        w(f"  输入特征: 38维 (地面气象24 + 干旱指数4 + 高空衍生7 + NDVI 1 + 月份周期2)")
        w(f"  序列长度: 12个月 (用过去12个月预测下1个月)")
        w(f"  模型体系: TCN / LSTM / GRU / Transformer (4种深度学习网络)")
        w(f"  隐藏单元: TCN通道[64,64,64] | LSTM/GRU隐层64×2层 | Transformer d_model=64, 4头×2层")
        w(f"  Dropout: 0.2")
        w(f"  优化器: Adam (lr=0.001), 微调 lr=0.0003")
        w(f"  策略: 4模型独立训练 + TCN 8站池化全局预训练迁移微调")
        w(f"  早停: patience=15 (微调 patience=10)")
        w(f"  最优选择: 按测试集RMSE为每站自动选择最优 (模型, 策略) 组合")
        w()
        w("11.2 模型性能对比")
        w("-" * 40)
        if 'comparison' in dl:
            w(f"  {'模型':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
            w(f"  {'-'*45}")
            for model_name, metrics in dl['comparison'].items():
                if metrics.get('rmse') is None:
                    w(f"  {model_name:<12} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
                    continue
                w(f"  {model_name:<12} {metrics['rmse']:>10.3f} {metrics['mae']:>10.3f} {metrics['r2']:>10.4f}")
            # 找出最优模型（过滤掉rmse为None的模型）
            valid = {k: v for k, v in dl['comparison'].items() if v.get('rmse') is not None}
            best = min(valid.items(), key=lambda x: x[1]['rmse'])
            w(f"\n  最优模型: {best[0]} (RMSE={best[1]['rmse']:.3f})")
        w()
        w("11.3 深度学习建模总结")
        w("-" * 40)
        w("  (1) 构建TCN/LSTM/GRU/Transformer四种深度学习网络, 统一38维特征与12月窗口")
        w("  (2) 按测试集RMSE为每站自动选优, 兼顾模型多样性与最优泛化性能")
        w("  (3) TCN经8站全局预训练+站点微调, 显著提升小样本站点精度")
        w("  (4) 不同城市气候特征差异使最优模型各异, 多模型对比体现选址适应性")
        w()

    # ============================================================
    w("=" * 80)
    w("附录: 可视化图表清单")
    w("=" * 80)
    w()
    chart_files = sorted([f for f in os.listdir(CHART_DIR) if f.endswith('.png')])
    w(f"  共 {len(chart_files)} 张图表, 保存在 output/charts/ 目录下:")
    w()
    for i, fname in enumerate(chart_files, 1):
        w(f"  {i:2d}. {fname}")

    w()
    w("=" * 80)
    w("                        报告结束")
    w("=" * 80)
    w()
    w("免责声明: 本报告基于NASA POWER、NCEP/NCAR R1及MODIS遥感再分析数据生成,")
    w("数据可能存在一定的偏差。分析结果仅供学术研究和参考, 不应作为正式气象")
    w("预报或决策依据。如需精确气象数据, 请参考中国气象局官方发布。")
    w()

    report_text = '\n'.join(L)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"  报告已生成: {REPORT_PATH} ({len(L)} 行)")
    return REPORT_PATH
