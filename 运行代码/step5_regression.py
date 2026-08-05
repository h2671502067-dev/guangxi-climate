"""
步骤5: 多元回归建模
===================
建立多元线性回归和随机森林回归模型, 分析气象变量间的定量关系。

数学方法:
  5.1 多元线性回归 (Ordinary Least Squares):
    模型: y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
    其中 ε ~ N(0, σ²)

    参数估计 (最小二乘法):
      β̂ = (X'X)⁻¹X'y

    模型评估:
      R² = 1 - SS_res / SS_tot = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²
      调整R² = 1 - (1-R²)(n-1)/(n-p-1)
      RMSE = √[(1/n) Σ(yᵢ - ŷᵢ)²]

    假设检验:
      F检验: H₀: β₁=β₂=...=βₚ=0
      t检验: H₀: βⱼ=0 (对每个系数)

  5.2 随机森林回归 (Random Forest Regressor):
    集成学习方法, 通过Bootstrap采样构建多棵决策树:
      1. 从训练集有放回抽样 n 个样本
      2. 对每个节点, 随机选择 m 个特征 (√p), 选择最优分裂
      3. 重复构建 B 棵树 (B=200)
      4. 预测: ŷ = (1/B) Σ f_b(x)

    特征重要性:
      Importance(xⱼ) = (1/B) Σ [impurity_before - impurity_after]

  5.3 多重共线性诊断 (VIF):
    方差膨胀因子:
      VIFⱼ = 1 / (1 - Rⱼ²)
    其中 Rⱼ² 为第 j 个特征对其余特征回归的 R²。
    VIF > 10 表示严重多重共线性, VIF > 5 需要关注。

  5.4 残差诊断:
    Shapiro-Wilk 正态检验:
      H₀: 残差服从正态分布
      p < 0.05 则拒绝正态性假设

    Breusch-Pagan 异方差检验:
      H₀: 残差方差为常数 (同方差)
      p < 0.05 则存在异方差

  5.5 岭回归 (Ridge Regression):
    在OLS损失函数中加入L2正则化项:
      β̂_ridge = argmin ||y - Xβ||² + α||β||²
    其中 α 为正则化强度, 通过交叉验证 (RidgeCV) 自动选择。
    岭回归可有效缓解多重共线性导致的系数估计不稳定问题。

  5.6 稳健标准误 OLS (Robust Standard Errors):
    使用 HC3 (异方差一致性) 估计量替代传统标准误:
      Var(β̂)_robust = (X'X)⁻¹ (Σ êᵢ²xᵢxᵢ') (X'X)⁻¹
    稳健标准误在异方差存在时仍能提供有效的统计推断。

  5.7 XGBoost 回归 (eXtreme Gradient Boosting):
    梯度提升决策树集成方法, 通过迭代添加新树来修正残差:
      F_m(x) = F_{m-1}(x) + η · h_m(x)
    其中 h_m(x) 为第 m 棵树, η 为学习率。
    目标函数: Obj = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₖ)
    正则化项: Ω(f) = γT + (λ/2)||w||²
    其中 T 为叶节点数, w 为叶权重。

  5.8 SHAP 模型解释 (SHapley Additive exPlanations):
    基于博弈论Shapley值的模型解释方法:
      φⱼ = Σ [|S|!(|N|-|S|-1)! / |N|!] · [f(S∪{j}) - f(S)]
    其中 N 为所有特征集合, S 为不包含特征 j 的子集。
    SHAP值量化了每个特征对模型预测的边际贡献。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from config import (REGRESSION_FEATURES, REGRESSION_TARGET, CHART_DIR,
                    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE,
                    REGRESSION_TEST_SIZE, REGRESSION_RANDOM_STATE,
                    plt_config)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

for k, v in plt_config.items():
    plt.rcParams[k] = v


def run(df, results):
    print("=" * 60)
    print("步骤5: 多元回归建模")
    print("=" * 60)

    X = df[REGRESSION_FEATURES].values
    y = df[REGRESSION_TARGET].values
    feature_names = REGRESSION_FEATURES

    # ==================== 数据划分 ====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=REGRESSION_TEST_SIZE, random_state=REGRESSION_RANDOM_STATE
    )
    print(f"  数据划分: 训练集 {len(y_train)} 条, 测试集 {len(y_test)} 条")

    # ==================== 5.1 多元线性回归 ====================
    print("  [5.1] 多元线性回归 (OLS)...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # 训练集评估
    y_pred_train_lr = lr.predict(X_train)
    r2_train_lr = r2_score(y_train, y_pred_train_lr)
    rmse_train_lr = np.sqrt(mean_squared_error(y_train, y_pred_train_lr))
    mae_train_lr = mean_absolute_error(y_train, y_pred_train_lr)
    n_train, p = X_train.shape
    adj_r2_lr = 1 - (1 - r2_train_lr) * (n_train - 1) / (n_train - p - 1)

    # 测试集评估
    y_pred_test_lr = lr.predict(X_test)
    r2_test_lr = r2_score(y_test, y_pred_test_lr)
    rmse_test_lr = np.sqrt(mean_squared_error(y_test, y_pred_test_lr))
    mae_test_lr = mean_absolute_error(y_test, y_pred_test_lr)

    # 交叉验证 (使用训练集)
    cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='r2')

    print(f"    [训练集] R²: {r2_train_lr:.4f}, 调整R²: {adj_r2_lr:.4f}, "
          f"RMSE: {rmse_train_lr:.3f}°C, MAE: {mae_train_lr:.3f}°C")
    print(f"    [测试集] R²: {r2_test_lr:.4f}, RMSE: {rmse_test_lr:.3f}°C, "
          f"MAE: {mae_test_lr:.3f}°C")
    print(f"    5折交叉验证 R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    截距: {lr.intercept_:.4f}")
    print(f"    回归系数:")
    for name, coef in zip(feature_names, lr.coef_):
        print(f"      {name}: {coef:+.4f}")

    # ==================== 5.2 随机森林回归 ====================
    print(f"\n  [5.2] 随机森林回归 (n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH})...")
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE
    )
    rf.fit(X_train, y_train)

    # 训练集评估
    y_pred_train_rf = rf.predict(X_train)
    r2_train_rf = r2_score(y_train, y_pred_train_rf)
    rmse_train_rf = np.sqrt(mean_squared_error(y_train, y_pred_train_rf))
    mae_train_rf = mean_absolute_error(y_train, y_pred_train_rf)

    # 测试集评估
    y_pred_test_rf = rf.predict(X_test)
    r2_test_rf = r2_score(y_test, y_pred_test_rf)
    rmse_test_rf = np.sqrt(mean_squared_error(y_test, y_pred_test_rf))
    mae_test_rf = mean_absolute_error(y_test, y_pred_test_rf)

    # 交叉验证 (使用训练集)
    cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')

    print(f"    [训练集] R²: {r2_train_rf:.4f}, RMSE: {rmse_train_rf:.3f}°C, "
          f"MAE: {mae_train_rf:.3f}°C")
    print(f"    [测试集] R²: {r2_test_rf:.4f}, RMSE: {rmse_test_rf:.3f}°C, "
          f"MAE: {mae_test_rf:.3f}°C")
    print(f"    5折交叉验证 R²: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")
    print(f"    特征重要性 (MDI):")
    importances = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1])
    for name, imp in importances:
        print(f"      {name}: {imp:.4f}")

    # ==================== 5.3 多重共线性诊断 (VIF) ====================
    print("\n  [5.3] 多重共线性诊断 (VIF)...")
    X_with_const = sm.add_constant(X_train)
    vif_data = pd.DataFrame()
    vif_data['特征'] = feature_names
    vif_data['VIF'] = [variance_inflation_factor(X_with_const, i + 1)
                       for i in range(len(feature_names))]
    for _, row in vif_data.iterrows():
        flag = " ***" if row['VIF'] > 10 else (" *" if row['VIF'] > 5 else "")
        print(f"    {row['特征']}: VIF = {row['VIF']:.2f}{flag}")

    # ==================== 5.4 残差诊断 ====================
    print("\n  [5.4] OLS残差诊断...")
    # 使用训练集残差进行诊断
    resid_lr = y_train - y_pred_train_lr

    # Shapiro-Wilk 正态检验
    shapiro_stat, shapiro_p = shapiro(resid_lr)
    print(f"    Shapiro-Wilk 正态检验: 统计量={shapiro_stat:.4f}, p值={shapiro_p:.4f}")
    if shapiro_p < 0.05:
        print(f"      -> p < 0.05, 残差不服从正态分布")
    else:
        print(f"      -> p >= 0.05, 不能拒绝正态性假设")

    # Breusch-Pagan 异方差检验
    bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(resid_lr, sm.add_constant(X_train))
    print(f"    Breusch-Pagan 异方差检验: LM统计量={bp_stat:.4f}, p值={bp_p:.4f}")
    if bp_p < 0.05:
        print(f"      -> p < 0.05, 存在异方差")
    else:
        print(f"      -> p >= 0.05, 不能拒绝同方差假设")

    # ==================== 5.5 岭回归 (处理多重共线性) ====================
    print("\n  [5.5] 岭回归 (处理多重共线性)...")
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.preprocessing import StandardScaler

    # 对全特征做岭回归
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 使用交叉验证选择最优alpha
    alphas = np.logspace(-3, 6, 50)
    ridge_cv = RidgeCV(alphas=alphas, cv=5)
    ridge_cv.fit(X_train_scaled, y_train)
    best_alpha = ridge_cv.alpha_

    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train_scaled, y_train)
    y_pred_test_ridge = ridge.predict(X_test_scaled)
    r2_test_ridge = r2_score(y_test, y_pred_test_ridge)
    rmse_test_ridge = np.sqrt(mean_squared_error(y_test, y_pred_test_ridge))

    print(f"    最优正则化参数 alpha = {best_alpha:.4f}")
    print(f"    [测试集] R²: {r2_test_ridge:.4f}, RMSE: {rmse_test_ridge:.3f}°C")
    print(f"    岭回归系数 (标准化后):")
    for name, coef in zip(feature_names, ridge.coef_):
        print(f"      {name}: {coef:+.4f}")

    # ==================== 5.6 稳健标准误 OLS (处理异方差) ====================
    print("\n  [5.6] 稳健标准误 OLS (处理异方差)...")
    # 使用statsmodels的OLS + HC3稳健标准误
    ols_model = sm.OLS(y_train, sm.add_constant(X_train))
    ols_robust = ols_model.fit(cov_type='HC3')
    # 打印稳健标准误下的系数和p值
    print("    稳健标准误 (HC3) 回归结果:")
    for name, coef, se, pval in zip(feature_names, ols_robust.params[1:],
                                    ols_robust.bse[1:], ols_robust.pvalues[1:]):
        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
        print(f"      {name}: coef={coef:+.4f}, SE={se:.4f}, p={pval:.4f} {sig}")

    # ==================== 5.7 XGBoost 回归 ====================
    xgb_results = {}
    if HAS_XGBOOST:
        print("\n  [5.7] XGBoost 回归...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RF_RANDOM_STATE,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)

        # 训练集评估
        y_pred_train_xgb = xgb_model.predict(X_train)
        r2_train_xgb = r2_score(y_train, y_pred_train_xgb)
        rmse_train_xgb = np.sqrt(mean_squared_error(y_train, y_pred_train_xgb))
        mae_train_xgb = mean_absolute_error(y_train, y_pred_train_xgb)

        # 测试集评估
        y_pred_test_xgb = xgb_model.predict(X_test)
        r2_test_xgb = r2_score(y_test, y_pred_test_xgb)
        rmse_test_xgb = np.sqrt(mean_squared_error(y_test, y_pred_test_xgb))
        mae_test_xgb = mean_absolute_error(y_test, y_pred_test_xgb)

        # 交叉验证 (使用训练集)
        cv_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')

        print(f"    [训练集] R²: {r2_train_xgb:.4f}, RMSE: {rmse_train_xgb:.3f}°C, "
              f"MAE: {mae_train_xgb:.3f}°C")
        print(f"    [测试集] R²: {r2_test_xgb:.4f}, RMSE: {rmse_test_xgb:.3f}°C, "
              f"MAE: {mae_test_xgb:.3f}°C")
        print(f"    5折交叉验证 R²: {cv_xgb.mean():.4f} ± {cv_xgb.std():.4f}")

        xgb_results = {
            'r2_train': r2_train_xgb, 'rmse_train': rmse_train_xgb, 'mae_train': mae_train_xgb,
            'r2_test': r2_test_xgb, 'rmse_test': rmse_test_xgb, 'mae_test': mae_test_xgb,
            'cv_mean': cv_xgb.mean(), 'cv_std': cv_xgb.std(),
            'importances': sorted(zip(feature_names, xgb_model.feature_importances_), key=lambda x: -x[1])
        }
    else:
        print("\n  [5.7] XGBoost 未安装, 跳过")

    # ==================== 5.8 SHAP 模型解释 ====================
    shap_values = None
    if HAS_SHAP and HAS_XGBOOST and xgb_results:
        print("\n  [5.8] SHAP 模型解释...")
        try:
            # 使用 TreeExplainer 计算 SHAP 值
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_train)

            # SHAP summary
            shap_mean = np.abs(shap_values).mean(axis=0)
            shap_importance = sorted(zip(feature_names, shap_mean), key=lambda x: -x[1])
            print("    SHAP 特征重要性 (平均|SHAP值|):")
            for name, val in shap_importance:
                print(f"      {name}: {val:.4f}")
        except Exception as e:
            print(f"    [警告] SHAP计算失败: {e}")
            shap_values = None
    elif not HAS_SHAP:
        print("\n  [5.8] SHAP 未安装, 跳过")

    # ==================== 5.9 可视化 ====================
    print("  [5.9] 绘制回归分析可视化图...")
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))

    # 线性回归: 实际 vs 预测 (测试集)
    ax = axes[0, 0]
    ax.scatter(y_test, y_pred_test_lr, alpha=0.5, s=20, c='#3498DB')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    ax.set_xlabel('实际值 (°C)')
    ax.set_ylabel('预测值 (°C)')
    ax.set_title(f'线性回归 (测试R²={r2_test_lr:.4f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 随机森林: 实际 vs 预测 (测试集)
    ax = axes[0, 1]
    ax.scatter(y_test, y_pred_test_rf, alpha=0.5, s=20, c='#2ECC71')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    ax.set_xlabel('实际值 (°C)')
    ax.set_ylabel('预测值 (°C)')
    ax.set_title(f'随机森林 (测试R²={r2_test_rf:.4f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 残差分布对比 (测试集)
    ax = axes[0, 2]
    resid_test_lr = y_test - y_pred_test_lr
    resid_test_rf = y_test - y_pred_test_rf
    ax.hist(resid_test_lr, bins=30, alpha=0.5, color='#3498DB', label='线性回归残差', density=True)
    ax.hist(resid_test_rf, bins=30, alpha=0.5, color='#2ECC71', label='随机森林残差', density=True)
    ax.set_title('残差分布对比 (测试集)', fontsize=12, fontweight='bold')
    ax.set_xlabel('残差 (°C)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 特征重要性
    ax = axes[1, 0]
    names_sorted = [x[0].replace('(', '\n(') for x in importances]
    imps_sorted = [x[1] for x in importances]
    ax.barh(range(len(names_sorted)), imps_sorted, color='#9B59B6', alpha=0.8)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.set_xlabel('重要性 (MDI)')
    ax.set_title('随机森林特征重要性', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 回归系数
    ax = axes[1, 1]
    coef_colors = ['#E74C3C' if c > 0 else '#3498DB' for c in lr.coef_]
    ax.barh(range(len(feature_names)), lr.coef_, color=coef_colors, alpha=0.8)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([f.replace('(', '\n(') for f in feature_names], fontsize=9)
    ax.set_xlabel('回归系数')
    ax.set_title('线性回归系数', fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')

    # 交叉验证对比
    ax = axes[1, 2]
    models = ['线性回归', '随机森林']
    cv_means = [cv_scores.mean(), cv_rf.mean()]
    cv_stds = [cv_scores.std(), cv_rf.std()]
    ax.bar(models, cv_means, yerr=cv_stds, color=['#3498DB', '#2ECC71'],
           alpha=0.8, capsize=5)
    ax.set_ylabel('R²')
    ax.set_title('5折交叉验证 R² 对比', fontsize=12, fontweight='bold')
    y_min = min(cv_means) - max(cv_stds) - 0.05
    y_max = max(cv_means) + max(cv_stds) + 0.05
    ax.set_ylim(max(0, y_min), y_max)
    ax.grid(True, alpha=0.3, axis='y')

    # 岭回归: 实际 vs 预测 (测试集)
    ax = axes[2, 0]
    ax.scatter(y_test, y_pred_test_ridge, alpha=0.5, s=20, c='#F39C12')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    ax.set_xlabel('实际值 (°C)')
    ax.set_ylabel('预测值 (°C)')
    ax.set_title(f'岭回归 (测试R²={r2_test_ridge:.4f}, α={best_alpha:.2f})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 所有模型测试集R²对比柱状图
    ax = axes[2, 1]
    model_names = ['OLS', 'Ridge', 'RF']
    r2_values = [r2_test_lr, r2_test_ridge, r2_test_rf]
    bar_colors = ['#3498DB', '#F39C12', '#2ECC71']
    bars = ax.bar(model_names, r2_values, color=bar_colors, alpha=0.8, width=0.6)
    for bar, val in zip(bars, r2_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('R²')
    ax.set_title('部分模型测试集 R² 对比', fontsize=12, fontweight='bold')
    ax.set_ylim(min(r2_values) - 0.05, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # 稳健标准误 vs 普通标准误对比
    ax = axes[2, 2]
    ols_params = dict(zip(feature_names, lr.coef_))
    robust_params = dict(zip(feature_names, ols_robust.params[1:]))
    robust_se = dict(zip(feature_names, ols_robust.bse[1:]))
    x_pos = np.arange(len(feature_names))
    width = 0.35
    ax.barh(x_pos - width / 2, list(ols_params.values()), width,
            color='#3498DB', alpha=0.8, label='OLS系数')
    ax.barh(x_pos + width / 2, list(robust_params.values()), width,
            color='#E74C3C', alpha=0.8, label='稳健标准误系数')
    # 添加误差线 (稳健标准误)
    ax.errorbar(list(robust_params.values()), x_pos + width / 2,
                xerr=list(robust_se.values()), fmt='none', ecolor='black', capsize=3)
    ax.set_yticks(x_pos)
    ax.set_yticklabels([f.replace('(', '\n(') for f in feature_names], fontsize=9)
    ax.set_xlabel('回归系数')
    ax.set_title('OLS vs 稳健标准误 (HC3) 系数对比', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    # XGBoost: 实际 vs 预测 (测试集)
    if xgb_results:
        ax = axes[0, 3]
        ax.scatter(y_test, y_pred_test_xgb, alpha=0.5, s=20, c='#E67E22')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        ax.set_xlabel('实际值 (°C)')
        ax.set_ylabel('预测值 (°C)')
        ax.set_title(f'XGBoost (测试R²={r2_test_xgb:.4f})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        axes[0, 3].text(0.5, 0.5, 'XGBoost\n未安装', ha='center', va='center',
                         fontsize=14, transform=axes[0, 3].transAxes, color='gray')

    # SHAP 特征重要性
    if shap_values is not None:
        ax = axes[1, 3]
        shap_mean = np.abs(shap_values).mean(axis=0)
        shap_idx = np.argsort(shap_mean)[::-1]
        ax.barh(range(len(feature_names)), shap_mean[shap_idx], color='#E74C3C', alpha=0.8)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i].replace('(', '\n(') for i in shap_idx], fontsize=9)
        ax.set_xlabel('平均 |SHAP值|')
        ax.set_title('SHAP 特征重要性', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    else:
        axes[1, 3].text(0.5, 0.5, 'SHAP\n未安装', ha='center', va='center',
                         fontsize=14, transform=axes[1, 3].transAxes, color='gray')

    # 全部模型测试集R²对比柱状图
    ax = axes[2, 3]
    all_models = ['OLS', 'Ridge', 'RF']
    all_r2 = [r2_test_lr, r2_test_ridge, r2_test_rf]
    all_colors = ['#3498DB', '#F39C12', '#2ECC71']
    if xgb_results:
        all_models.append('XGBoost')
        all_r2.append(r2_test_xgb)
        all_colors.append('#E67E22')
    bars = ax.bar(all_models, all_r2, color=all_colors, alpha=0.8, width=0.6)
    for bar, val in zip(bars, all_r2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('R²')
    ax.set_title('全部模型测试集 R² 对比', fontsize=12, fontweight='bold')
    ax.set_ylim(min(all_r2) - 0.05, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('广西气温回归建模分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{CHART_DIR}/16_回归建模分析.png')
    plt.close()

    if shap_values is not None:
        try:
            fig_shap = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_train, feature_names=feature_names,
                              show=False, max_display=len(feature_names))
            plt.tight_layout()
            plt.savefig(f'{CHART_DIR}/29_SHAP特征解释.png')
            plt.close()
        except Exception:
            pass

    results['regression'] = {
        'lr': {'r2': r2_train_lr, 'adj_r2': adj_r2_lr, 'rmse': rmse_train_lr,
               'mae': mae_train_lr, 'test_r2': r2_test_lr, 'test_rmse': rmse_test_lr,
               'test_mae': mae_test_lr,
               'coef': dict(zip(feature_names, lr.coef_)), 'intercept': lr.intercept_,
               'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()},
        'rf': {'r2': r2_train_rf, 'rmse': rmse_train_rf, 'mae': mae_train_rf,
               'test_r2': r2_test_rf, 'test_rmse': rmse_test_rf, 'test_mae': mae_test_rf,
               'importances': importances, 'cv_mean': cv_rf.mean(), 'cv_std': cv_rf.std()},
        'ridge': {'r2_test': float(r2_test_ridge), 'rmse_test': float(rmse_test_ridge),
                  'best_alpha': float(best_alpha),
                  'coef': dict(zip(feature_names, ridge.coef_))},
        'ols_robust': {'params': dict(zip(feature_names, ols_robust.params[1:].tolist())),
                       'pvalues': dict(zip(feature_names, ols_robust.pvalues[1:].tolist())),
                       'bse': dict(zip(feature_names, ols_robust.bse[1:].tolist()))},
        'vif': vif_data.to_dict('records'),
        'shapiro': {'statistic': float(shapiro_stat), 'p_value': float(shapiro_p)},
        'breuschpagan': {'lm_statistic': float(bp_stat), 'p_value': float(bp_p)},
    }
    if xgb_results:
        results['regression']['xgboost'] = xgb_results
    if shap_values is not None:
        results['regression']['shap_importance'] = shap_importance
    n_charts = 1
    if shap_values is not None:
        n_charts = 2
    print(f"  步骤5完成, 共生成{n_charts}张图表。\n")
    return results
