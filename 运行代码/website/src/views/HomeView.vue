<script setup>
import { onMounted, ref } from 'vue'
import overview from '../data/overview.json'

const assetsBase = document.baseURI.replace(/#.*$/, '')

const stats = ref([
  { num: overview.n_samples, label: '月度观测' },
  { num: overview.stations.length, label: '代表城市' },
  { num: overview.n_features, label: '维特征' },
  { num: '4', label: '深度学习模型' },
])

const modules = [
  { no: '01', name: '数据加载与清洗', desc: '广西8市全省平均 · 单位换算 · 缺失值插值' },
  { no: '02', name: '描述性统计', desc: '均值/标准差/CV · 箱线图 · 季节对比' },
  { no: '03', name: '相关性分析', desc: 'Pearson相关矩阵 · 显著性检验' },
  { no: '04', name: '时间序列分解', desc: '加法分解 · ADF平稳性检验' },
  { no: '05', name: '时序预测', desc: 'SARIMA网格搜索 · Holt-Winters · 残差诊断' },
  { no: '06', name: '多元回归', desc: 'OLS/Ridge/RF/XGBoost · SHAP · VIF · HC3' },
  { no: '07', name: '聚类分析', desc: 'K-Means · PCA降维 · 层次聚类' },
  { no: '08', name: '趋势检验', desc: 'Mann-Kendall · Sen斜率估计' },
  { no: '09', name: '极端事件', desc: '百分位法 · Z-score异常检测' },
  { no: '10', name: '深度学习', desc: 'TCN/LSTM/GRU/Transformer · 迁移学习' },
]

onMounted(() => {
  const io = new IntersectionObserver(
    (es) => es.forEach((e) => e.isIntersecting && (e.target.classList.add('in'), io.unobserve(e.target))),
    { threshold: 0.12 }
  )
  document.querySelectorAll('.reveal').forEach((el) => io.observe(el))
})
</script>

<template>
  <div>
    <!-- Hero -->
    <section class="relative flex min-h-[88svh] items-center overflow-hidden">
      <div class="absolute inset-0 bg-cover bg-center" :style="{ backgroundImage: 'url(' + assetsBase + 'hero.png)' }"></div>
      <div class="absolute inset-0 bg-gradient-to-b from-white/20 via-white/35 to-ink"></div>
      <div class="relative z-10 mx-auto w-full max-w-6xl px-6" style="transform: translateY(-40px);">
        <p class="mb-5 text-xs font-semibold tracking-[0.28em] text-sky">2005—2025 · 多源气象数据 · 数学建模与深度学习</p>
        <h1 class="max-w-3xl text-5xl font-black leading-tight md:text-7xl">
          广西壮族自治区<br /><span class="text-sky">气象数据建模分析</span>
        </h1>
        <p class="mt-6 max-w-xl text-base text-ice/80 md:text-lg">
          8 个代表城市 · 38 维气象特征 · 10 个分析模块 —— 从描述统计到 TCN / LSTM / GRU / Transformer 多模型深度学习预测
        </p>
        <router-link
          to="/dashboard"
          class="mt-9 inline-block rounded-full bg-sky px-8 py-3.5 text-sm font-semibold tracking-widest text-white shadow-lg shadow-sky/30 transition-all hover:bg-[#1e90e0]"
        >开始探索 →</router-link>
      </div>
      <!-- 核心数据条 -->
      <div class="absolute bottom-0 left-0 right-0 z-10 border-t border-line bg-white/70 backdrop-blur-md">
        <div class="mx-auto grid max-w-6xl grid-cols-2 divide-x divide-line md:grid-cols-4">
          <div v-for="s in stats" :key="s.label" class="px-6 py-6 text-center">
            <div class="text-3xl font-black text-sky md:text-4xl">{{ s.num }}</div>
            <div class="mt-1 text-xs tracking-widest text-dim">{{ s.label }}</div>
          </div>
        </div>
      </div>
    </section>

    <!-- 数据来源 -->
    <section class="border-t border-line bg-ink-alt px-6 py-20">
      <div class="mx-auto max-w-6xl">
        <p class="mb-3 text-xs font-semibold tracking-[0.26em] text-sky">DATA SOURCES · 数据来源</p>
        <h2 class="mb-10 text-3xl font-bold md:text-4xl">多源数据，同一套分析管道</h2>
        <div class="grid gap-6 md:grid-cols-3">
          <div class="rounded-xl border border-line bg-surface p-7 shadow-sm">
            <h3 class="text-lg font-bold">地面气象</h3>
            <p class="mt-2 text-sm text-dim">NASA POWER · 24 项<br>气温 / 降水 / 湿度 / 辐射 / 土壤</p>
          </div>
          <div class="rounded-xl border border-line bg-surface p-7 shadow-sm">
            <h3 class="text-lg font-bold">高空再分析</h3>
            <p class="mt-2 text-sm text-dim">NCEP/NCAR R1 · 7 项<br>K 指数 / 温度平流 / 水汽通量</p>
          </div>
          <div class="rounded-xl border border-line bg-surface p-7 shadow-sm">
            <h3 class="text-lg font-bold">植被与干旱</h3>
            <p class="mt-2 text-sm text-dim">MODIS NDVI · SPI/SPEI<br>植被指数 + 干旱指数 + 月份周期</p>
          </div>
        </div>
      </div>
    </section>

    <!-- 模块导航 -->
    <section class="px-6 py-20">
      <div class="mx-auto max-w-6xl">
        <p class="mb-3 text-xs font-semibold tracking-[0.26em] text-sky">PIPELINE · 分析流程</p>
        <h2 class="mb-4 text-3xl font-bold md:text-4xl">十个模块，一条完整链路</h2>
        <p class="mb-10 max-w-2xl text-sm text-dim">覆盖数据探索、统计检验、时序建模到深度学习预测的完整分析流程。</p>
        <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <div v-for="m in modules" :key="m.no" class="reveal rounded-xl border border-line bg-surface p-6 shadow-sm">
            <span class="text-xs font-bold tracking-widest text-amber">{{ m.no }}</span>
            <h3 class="mt-2 text-lg font-bold">{{ m.name }}</h3>
            <p class="mt-2 text-sm text-dim">{{ m.desc }}</p>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>
