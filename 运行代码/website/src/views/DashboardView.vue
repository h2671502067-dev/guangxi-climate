<script setup>
import { onMounted, ref, computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, DataZoomComponent } from 'echarts/components'
import models from '../data/models.json'
import overview from '../data/overview.json'
import timeseries from '../data/timeseries.json'

const baseUrl = document.baseURI.replace(/#.*$/, '')

use([CanvasRenderer, LineChart, BarChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent])

const cities = overview.stations
const selectedCity = ref('南宁')

// ===================== 变量控制按钮 =====================
const varGroups = [
  {
    name: '温度',
    vars: [
      { key: 'temp', label: '平均气温', unit: '°C', color: '#ff9848' },
      { key: 'tmax', label: '最高气温', unit: '°C', color: '#ff9848' },
      { key: 'tmin', label: '最低气温', unit: '°C', color: '#ff9848' },
      { key: 'dew', label: '露点温度', unit: '°C', color: '#ff9848' },
    ],
  },
  {
    name: '降水/湿度',
    vars: [
      { key: 'precip', label: '降水量', unit: 'mm/月', color: '#42a5f5', agg: 'sum' },
      { key: 'humidity', label: '相对湿度', unit: '%', color: '#42a5f5' },
    ],
  },
  {
    name: '风/气压',
    vars: [
      { key: 'wind', label: '风速', unit: 'm/s', color: '#5ccfe6' },
      { key: 'pressure', label: '气压', unit: 'kPa', color: '#5ccfe6' },
    ],
  },
  {
    name: '能量',
    vars: [
      { key: 'radiation', label: '短波辐射', unit: 'kWh/m²·日', color: '#a78bfa' },
      { key: 'sunshine', label: '日照时数', unit: 'h/日', color: '#a78bfa' },
      { key: 'evap', label: '蒸散发', unit: 'mm/日', color: '#a78bfa' },
      { key: 'cloud', label: '总云量', unit: '%', color: '#a78bfa' },
    ],
  },
  {
    name: '生态',
    vars: [
      { key: 'soil', label: '根区土壤湿度', unit: '', color: '#4ade80' },
      { key: 'ndvi', label: 'NDVI 植被指数', unit: '', color: '#4ade80' },
    ],
  },
  {
    name: '干旱',
    vars: [
      { key: 'spi1', label: 'SPI-1', unit: '', color: '#fbbf24' },
      { key: 'spi3', label: 'SPI-3', unit: '', color: '#fbbf24' },
      { key: 'spei1', label: 'SPEI-1', unit: '', color: '#fbbf24' },
      { key: 'spei3', label: 'SPEI-3', unit: '', color: '#fbbf24' },
    ],
  },
]
const varMap = Object.fromEntries(varGroups.flatMap((g) => g.vars.map((v) => [v.key, v])))
const selectedVar = ref('temp')

// ===================== 粒度 / 图表类型按钮 =====================
const granularities = [
  { key: 'month', label: '月度' },
  { key: 'quarter', label: '季度' },
  { key: 'year', label: '年度' },
]
const selectedGranularity = ref('month')

const chartTypes = [
  { key: 'line', label: '折线' },
  { key: 'bar', label: '柱状' },
  { key: 'area', label: '面积' },
]
const selectedType = ref('line')

// ===================== 数据聚合: 月度原样, 季度/年度取均值(降水求和) =====================
const raw = computed(() => timeseries[selectedCity.value] || [])

const seriesData = computed(() => {
  const v = varMap[selectedVar.value]
  const agg = v.agg || 'mean'
  if (selectedGranularity.value === 'month') {
    return {
      labels: raw.value.map((x) => x.date),
      values: raw.value.map((x) => x[selectedVar.value]),
    }
  }
  const buckets = new Map()
  for (const x of raw.value) {
    const [y, m] = x.date.split('-').map(Number)
    const key = selectedGranularity.value === 'year'
      ? `${y}`
      : `${y}-Q${Math.floor((m - 1) / 3) + 1}`
    if (!buckets.has(key)) buckets.set(key, [])
    buckets.get(key).push(x[selectedVar.value])
  }
  const keys = [...buckets.keys()].sort()
  return {
    labels: keys,
    values: keys.map((k) => {
      const arr = buckets.get(k).filter((x) => x != null)
      if (!arr.length) return null
      return agg === 'sum'
        ? arr.reduce((a, b) => a + b, 0)
        : arr.reduce((a, b) => a + b, 0) / arr.length
    }),
  }
})

const mainOption = computed(() => {
  const v = varMap[selectedVar.value]
  const { labels, values } = seriesData.value
  const isBar = selectedType.value === 'bar'
  const isArea = selectedType.value === 'area'
  const isMonth = selectedGranularity.value === 'month'
  const color = v.color
  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      valueFormatter: (val) => (val == null ? '-' : `${Number(val).toFixed(2)}${v.unit ? ' ' + v.unit : ''}`),
    },
    legend: { data: [v.label], textStyle: { color: '#64748b' }, top: 0 },
    grid: { left: 60, right: 24, top: 44, bottom: isMonth ? 60 : 30 },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { color: '#64748b', interval: isMonth ? 11 : 0 },
      axisLine: { lineStyle: { color: 'rgba(30,41,59,0.15)' } },
    },
    yAxis: {
      type: 'value',
      name: v.unit,
      nameTextStyle: { color: '#64748b' },
      axisLabel: { color: '#64748b' },
      splitLine: { lineStyle: { color: 'rgba(30,41,59,0.08)' } },
    },
    dataZoom: isMonth
      ? [
          { type: 'inside', start: 0, end: 100 },
          { type: 'slider', height: 20, bottom: 6 },
        ]
      : undefined,
    series: [
      {
        name: v.label,
        type: isBar ? 'bar' : 'line',
        showSymbol: false,
        barMaxWidth: isBar ? (isMonth ? 4 : 24) : undefined,
        data: values,
        itemStyle: isBar ? { color, borderRadius: [4, 4, 0, 0] } : { color },
        lineStyle: { width: 1.8, color },
        areaStyle: isArea
          ? {
              color: {
                type: 'linear',
                x: 0, y: 0, x2: 0, y2: 1,
                colorStops: [
                  { offset: 0, color: `${color}40` },
                  { offset: 1, color: `${color}05` },
                ],
              },
            }
          : undefined,
      },
    ],
  }
})

// 模型对比柱状图 (4模型独立 + TCN迁移)
const modelCompareOption = computed(() => ({
  backgroundColor: 'transparent',
  tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
  legend: { data: ['独立训练', '迁移学习'], textStyle: { color: '#64748b' }, top: 0 },
  grid: { left: 48, right: 16, top: 40, bottom: 44 },
  xAxis: { type: 'category', data: models.station_cn, axisLabel: { color: '#64748b', rotate: 30 }, axisLine: { lineStyle: { color: 'rgba(30,41,59,0.15)' } } },
  yAxis: { type: 'value', name: 'RMSE (°C)', nameTextStyle: { color: '#64748b' }, axisLabel: { color: '#64748b' }, splitLine: { lineStyle: { color: 'rgba(30,41,59,0.08)' } } },
  series: [
    { name: '独立训练', type: 'bar', barGap: '-100%', itemStyle: { color: 'rgba(66,165,245,0.45)' }, data: models.tcn_ind },
    { name: '迁移学习', type: 'bar', itemStyle: { color: '#ff9848' }, data: models.tcn_transfer },
  ],
}))

// 各模型8市平均RMSE
const modelAvgOption = computed(() => ({
  backgroundColor: 'transparent',
  tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
  grid: { left: 56, right: 20, top: 20, bottom: 30 },
  xAxis: { type: 'value', axisLabel: { color: '#64748b' }, splitLine: { lineStyle: { color: 'rgba(30,41,59,0.08)' } } },
  yAxis: { type: 'category', data: models.models.map((m) => m.name).reverse(), axisLabel: { color: '#64748b' }, axisLine: { lineStyle: { color: 'rgba(30,41,59,0.15)' } } },
  series: [{ type: 'bar', barWidth: 18, itemStyle: { color: (p) => ['#42a5f5', '#ff9848', '#5ccfe6', '#a78bfa'][p.dataIndex] }, label: { show: true, position: 'right', color: '#1e293b', fontSize: 12 }, data: models.models.map((m) => m.avg_rmse).reverse() }],
}))

// 静态图表分组 (覆盖全部28张)
const chartGroups = [
  { title: '基础描述统计', charts: ['01_核心气象要素时间序列', '02_气温月度箱线图', '03_降水量月度箱线图', '04_年度气象指标趋势'] },
  { title: '热力图与季节', charts: ['05_气温热力图', '06_降水热力图', '07_季节性对比', '08_年际滑动平均趋势'] },
  { title: '相关性与时序分解', charts: ['09_相关性矩阵', '10_关键变量散点图', '11_气温时间序列分解', '12_降水时间序列分解'] },
  { title: '预测建模', charts: ['13_SARIMA气温预测', '14_HoltWinters降水预测', '15_SARIMA残差诊断', '16_回归建模分析'] },
  { title: '聚类与趋势', charts: ['17_聚类分析', '18_MannKendall趋势检验', '19_极端事件检测', '20_风速风向分析'] },
  { title: '土壤与辐射', charts: ['21_土壤温湿度分析', '22_辐射日照蒸散分析', '24_湿度分析', '25_气压分析'] },
  { title: '环境与深度学习', charts: ['26_气候特征雷达图', '27_深度学习预测对比', '28_模型系统对比', '29_SHAP特征解释'] },
]

const lightbox = ref(null)

function openLightbox(src) {
  lightbox.value = src
}
function closeLightbox() {
  lightbox.value = null
}

onMounted(() => {
  const io = new IntersectionObserver(
    (es) => es.forEach((e) => e.isIntersecting && (e.target.classList.add('in'), io.unobserve(e.target))),
    { threshold: 0.1 }
  )
  document.querySelectorAll('.reveal').forEach((el) => io.observe(el))
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeLightbox()
  })
})
</script>

<template>
  <div class="px-6 py-16">
    <div class="mx-auto max-w-6xl">
      <p class="mb-3 text-xs font-semibold tracking-[0.26em] text-sky">DASHBOARD · 数据分析</p>
      <h1 class="text-3xl font-bold md:text-4xl">动态图表与全量分析结果</h1>
      <p class="mt-3 max-w-2xl text-sm text-dim">18 项气象变量交互探索 · 月度/季度/年度视图 · 模型对比 · 28 张完整分析图表。</p>

      <!-- 动态图表 -->
      <section class="mt-12">
        <h2 class="mb-5 text-xl font-bold">城市气象变量探索（交互）</h2>

        <!-- 城市选择 -->
        <div class="mb-5 flex flex-wrap gap-2">
          <button
            v-for="c in cities"
            :key="c"
            class="rounded-full border px-4 py-1.5 text-xs transition-all"
            :class="selectedCity === c ? 'border-sky bg-sky/15 text-sky' : 'border-line text-dim hover:text-ice'"
            @click="selectedCity = c"
          >{{ c }}</button>
        </div>

        <!-- 变量分组按钮 -->
        <div class="space-y-2">
          <div v-for="g in varGroups" :key="g.name" class="flex flex-wrap items-center gap-2">
            <span class="w-16 shrink-0 text-xs font-semibold tracking-wider text-dim">{{ g.name }}</span>
            <button
              v-for="v in g.vars"
              :key="v.key"
              class="rounded-full border px-3.5 py-1.5 text-xs transition-all"
              :class="selectedVar === v.key
                ? 'border-sky bg-sky/15 font-semibold text-sky'
                : 'border-line text-dim hover:border-sky/40 hover:text-ice'"
              @click="selectedVar = v.key"
            >{{ v.label }}</button>
          </div>
        </div>

        <!-- 粒度 + 图表类型 -->
        <div class="mt-5 flex flex-wrap items-center gap-x-8 gap-y-3">
          <div class="flex items-center gap-2">
            <span class="text-xs text-dim">时间粒度</span>
            <button
              v-for="g in granularities"
              :key="g.key"
              class="rounded-md border px-3 py-1.5 text-xs transition-all"
              :class="selectedGranularity === g.key ? 'border-sky bg-sky/15 font-semibold text-sky' : 'border-line text-dim hover:text-ice'"
              @click="selectedGranularity = g.key"
            >{{ g.label }}</button>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-xs text-dim">图表样式</span>
            <button
              v-for="t in chartTypes"
              :key="t.key"
              class="rounded-md border px-3 py-1.5 text-xs transition-all"
              :class="selectedType === t.key ? 'border-sky bg-sky/15 font-semibold text-sky' : 'border-line text-dim hover:text-ice'"
              @click="selectedType = t.key"
            >{{ t.label }}</button>
          </div>
        </div>

        <!-- 主图表 -->
        <div class="mt-5 rounded-xl border border-line bg-surface p-4">
          <div class="mb-1 flex flex-wrap items-baseline justify-between gap-2">
            <h3 class="text-sm font-bold">{{ varMap[selectedVar].label }} · {{ selectedCity }} · {{ granularities.find((g) => g.key === selectedGranularity).label }}</h3>
            <span class="text-xs text-dim">单位：{{ varMap[selectedVar].unit || '—' }}</span>
          </div>
          <div style="height: 380px; width: 100%;">
            <VChart :option="mainOption" autoresize style="height: 100%; width: 100%;" />
          </div>
        </div>
      </section>

      <!-- 模型对比 -->
      <section class="mt-14">
        <h2 class="mb-5 text-xl font-bold">TCN 独立训练 vs 迁移学习</h2>
        <div class="grid gap-6 lg:grid-cols-2">
          <div class="rounded-xl border border-line bg-surface p-4">
            <div style="height: 340px; width: 100%;">
              <VChart :option="modelCompareOption" autoresize style="height: 100%; width: 100%;" />
            </div>
          </div>
          <div class="rounded-xl border border-line bg-surface p-4">
            <div style="height: 340px; width: 100%;">
              <VChart :option="modelAvgOption" autoresize style="height: 100%; width: 100%;" />
            </div>
          </div>
        </div>
      </section>

      <!-- 静态图表 -->
      <section v-for="(g, gi) in chartGroups" :key="gi" class="mt-14">
        <h2 class="mb-5 border-l-2 border-sky pl-3 text-lg font-bold">{{ g.title }}</h2>
        <div class="grid gap-4 md:grid-cols-2">
          <figure
            v-for="c in g.charts"
            :key="c"
            class="reveal group cursor-zoom-in overflow-hidden rounded-xl border border-line bg-surface"
            @click="openLightbox(baseUrl + 'charts/' + c + '.png')"
          >
            <img :src="baseUrl + 'charts/' + c + '.png'" :alt="c" loading="lazy" class="w-full transition-transform duration-500 group-hover:scale-105" />
          </figure>
        </div>
      </section>
    </div>

    <!-- 灯箱 -->
    <div
      v-if="lightbox"
      class="fixed inset-0 z-50 flex cursor-zoom-out items-center justify-center bg-black/90 p-6"
      @click="closeLightbox"
    >
      <img :src="lightbox" class="max-h-[88vh] max-w-full rounded-lg shadow-2xl" />
    </div>
  </div>
</template>
