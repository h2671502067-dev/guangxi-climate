<script setup>
import { onMounted } from 'vue'

const methods = [
  {
    no: '01', name: '数据加载与清洗',
    desc: '广西8市逐月数据合并，单位换算后映射为标准Schema，缺失值线性插值+边界填充。',
    formula: 'x_t = x_{t-1} + (x_{t+1} - x_{t-1}) · (t - t_{t-1}) / (t_{t+1} - t_{t-1})',
  },
  {
    no: '02', name: '描述性统计',
    desc: '计算均值、标准差、偏度、峰度与变异系数，评估各气象要素的分布形态。',
    formula: 'CV = (s / x̄) × 100%,  Skew = E[(X-μ)³]/σ³',
  },
  {
    no: '03', name: '相关性分析',
    desc: 'Pearson相关系数刻画变量间线性关联，|r|>0.8 视为强相关。',
    formula: 'r = Σ(xᵢ-x̄)(yᵢ-ȳ) / √[Σ(xᵢ-x̄)² · Σ(yᵢ-ȳ)²]',
  },
  {
    no: '04', name: '时间序列分解',
    desc: '加法分解将序列拆为趋势、季节与残差分量，ADF检验判断平稳性。',
    formula: 'X(t) = T(t) + S(t) + R(t),  Δyₜ = α + βt + γyₜ₋₁ + ΣδᵢΔyₜ₋ᵢ + εₜ',
  },
  {
    no: '05', name: 'SARIMA / Holt-Winters 预测',
    desc: 'SARIMA网格搜索以AIC选参，Ljung-Box检验残差白噪声并自动修正自相关。',
    formula: 'Φ(B¹²)φ(B)(1-B)^d(1-B¹²)^D yₜ = Θ(B¹²)θ(B) εₜ',
  },
  {
    no: '06', name: '多元回归与解释',
    desc: 'OLS + VIF共线性诊断 + 岭回归 + 随机森林 + XGBoost，SHAP值解释模型决策。',
    formula: 'y = β₀ + β₁x₁ + … + βₚxₚ + ε,  φⱼ = Σ[|S|!(|N|-|S|-1)!/|N|!]·[f(S∪{j}) - f(S)]',
  },
  {
    no: '07', name: '聚类分析',
    desc: 'K-Means按肘部法则与轮廓系数选K，PCA降维可视化，Ward层次聚类验证。',
    formula: 'min Σₖ Σ_{xᵢ∈Cₖ} ||xᵢ - μₖ||²,  Z = XV (协方差特征分解)',
  },
  {
    no: '08', name: 'Mann-Kendall 趋势检验',
    desc: '非参数秩检验判断趋势显著性，Sen斜率估计趋势强度，含ties修正。',
    formula: 'S = Σ_{i<j} sgn(xⱼ-xᵢ),  Z = (S±1)/√Var(S),  β = median{(xⱼ-xᵢ)/(j-i)}',
  },
  {
    no: '09', name: '极端事件检测',
    desc: '百分位法设定极端阈值，Z-score识别统计异常值。',
    formula: '阈值 = P95 / P05,  Z = (x-μ)/σ,  |Z| > 2 判为异常',
  },
  {
    no: '10', name: '深度学习预测',
    desc: 'TCN/LSTM/GRU/Transformer 四种网络对比，TCN另经8站池化预训练+迁移微调，按测试RMSE自动选优。',
    formula: 'TCN: yₜ = Σf_k·x_{t-k} (膨胀因果卷积)  |  LSTM: fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ',
  },
]

const validation = [
  { name: '时序滚动划分', desc: '前228个月训练，后24个月(2024-2025)测试，杜绝未来信息泄漏' },
  { name: '5折交叉验证', desc: '回归模型5折CV取均值±标准差，评估泛化稳定性' },
  { name: '内部验证集早停', desc: '训练窗口末尾12个窗口作验证，损失连续patience轮不降即停' },
  { name: '测试RMSE自动选优', desc: '每站从4模型独立+TCN迁移共9个候选中选RMSE最小组合' },
]

onMounted(() => {
  const io = new IntersectionObserver(
    (es) => es.forEach((e) => e.isIntersecting && (e.target.classList.add('in'), io.unobserve(e.target))),
    { threshold: 0.1 }
  )
  document.querySelectorAll('.reveal').forEach((el) => io.observe(el))
})
</script>

<template>
  <div class="px-6 py-16">
    <div class="mx-auto max-w-4xl">
      <p class="mb-3 text-xs font-semibold tracking-[0.26em] text-sky">METHODS · 方法论</p>
      <h1 class="text-3xl font-bold md:text-4xl">分析方法的数学基础</h1>
      <p class="mt-3 max-w-2xl text-sm text-dim">十个模块的核心方法与数学表达。</p>

      <div class="mt-12 space-y-6">
        <article
          v-for="m in methods"
          :key="m.no"
          class="reveal border border-line bg-surface p-7"
        >
          <div class="flex items-baseline gap-4">
            <span class="text-sm font-bold tracking-widest text-amber">{{ m.no }}</span>
            <h2 class="text-lg font-bold">{{ m.name }}</h2>
          </div>
          <p class="mt-3 text-sm text-dim">{{ m.desc }}</p>
          <code class="mt-4 block overflow-x-auto rounded-md border border-line bg-ink px-4 py-3 text-sm text-sky">
            {{ m.formula }}
          </code>
        </article>
      </div>

      <!-- 验证方法 -->
      <section class="mt-16">
        <h2 class="mb-6 text-xl font-bold">模型验证方式</h2>
        <div class="grid gap-4 md:grid-cols-2">
          <div v-for="v in validation" :key="v.name" class="reveal border-l-2 border-sky bg-surface p-6">
            <h3 class="font-bold">{{ v.name }}</h3>
            <p class="mt-2 text-sm text-dim">{{ v.desc }}</p>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>
