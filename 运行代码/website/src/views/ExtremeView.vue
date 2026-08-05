<script setup>
import { onMounted, ref, computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { HeatmapChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, VisualMapComponent } from 'echarts/components'
import overview from '../data/overview.json'
import timeseries from '../data/timeseries.json'

use([CanvasRenderer, HeatmapChart, GridComponent, TooltipComponent, VisualMapComponent])

const cities = overview.stations
const selectedCity = ref('南宁')

const metrics = [
  { key: 'heat', label: '极端高温', col: 'temp', dir: 'high', note: '月均气温 ≥ P95', palette: ['#3b82f6', '#f8fafc', '#ef4444'] },
  { key: 'cold', label: '极端低温', col: 'tmin', dir: 'low', note: '最低气温 ≤ P05', palette: ['#3b82f6', '#f8fafc', '#ef4444'] },
  { key: 'wet', label: '极端多雨', col: 'precip', dir: 'high', note: '降水量 ≥ P95', palette: ['#f59e0b', '#f8fafc', '#3b82f6'] },
  { key: 'dry', label: '极端少雨', col: 'precip', dir: 'low', note: '降水量 ≤ P05', palette: ['#f59e0b', '#f8fafc', '#3b82f6'] },
]
const selectedMetric = ref('heat')
const metric = computed(() => metrics.find((m) => m.key === selectedMetric.value))

const raw = computed(() => timeseries[selectedCity.value] || [])

function percentile(arr, q) {
  const s = [...arr].filter((x) => x != null).sort((a, b) => a - b)
  if (!s.length) return 0
  const idx = Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))
  return s[idx]
}

// 年 × 月 z-score 热力图
const calendarData = computed(() => {
  const d = raw.value
  const vals = d.map((x) => x[metric.value.col])
  const mu = vals.reduce((a, b) => a + b, 0) / vals.length
  const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mu) ** 2, 0) / vals.length) || 1
  const years = []
  for (const x of d) {
    const [y, m] = x.date.split('-').map(Number)
    if (!years.includes(y)) years.push(y)
  }
  years.sort((a, b) => b - a)
  return { years, data: d.map((x) => {
    const [y, m] = x.date.split('-').map(Number)
    const z = Math.round(((x[metric.value.col] - mu) / sd) * 100) / 100
    return {
      value: [m - 1, years.indexOf(y), z],
      label: { show: true, formatter: z.toFixed(1), fontSize: 10, color: Math.abs(z) > 1.5 ? '#ffffff' : '#1e293b' },
    }
  }) }
})

const heatmapOption = computed(() => {
  const { years, data } = calendarData.value
  return {
    backgroundColor: 'transparent',
    tooltip: {
      position: 'top',
      formatter: (p) => {
        const m = p.value[0] + 1
        const y = years[p.value[1]]
        const rec = raw.value.find((x) => x.date === `${y}-${String(m).padStart(2, '0')}`)
        return `${y}年${m}月<br/>${metric.value.col === 'precip' ? '降水量' : metric.value.col === 'tmin' ? '最低气温' : '月均气温'}：${rec ? rec[metric.value.col] : '-'}<br/>z = <b>${p.value[2]}</b>`
      },
    },
    grid: { left: 56, right: 30, top: 16, bottom: 84 },
    xAxis: {
      type: 'category',
      data: Array.from({ length: 12 }, (_, i) => `${i + 1}月`),
      axisLabel: { color: '#64748b' },
      axisLine: { lineStyle: { color: 'rgba(30,41,59,0.15)' } },
    },
    yAxis: {
      type: 'category',
      data: years,
      axisLabel: { color: '#64748b' },
      axisLine: { lineStyle: { color: 'rgba(30,41,59,0.15)' } },
    },
    visualMap: {
      min: -3, max: 3,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 10,
      itemWidth: 12, itemHeight: 120,
      textStyle: { color: '#64748b' },
      inRange: { color: metric.value.palette },
    },
    series: [
      {
        type: 'heatmap',
        data,
        emphasis: {
          itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.4)' },
        },
      },
    ],
  }
})

// 统计卡片
const stats = computed(() => {
  const d = raw.value
  const col = metric.value.col
  const vals = d.map((x) => x[col])
  const mu = vals.reduce((a, b) => a + b, 0) / vals.length
  const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mu) ** 2, 0) / vals.length) || 1
  const thr = percentile(vals, metric.value.dir === 'high' ? 0.95 : 0.05)
  const extreme = vals.map((v, i) => ({ v, z: (v - mu) / sd, i }))
    .filter((x) => metric.value.dir === 'high' ? x.v >= thr : x.v <= thr)
  let worst = null
  d.forEach((x, i) => {
    const z = (x[col] - mu) / sd
    if (!worst || Math.abs(z) > Math.abs(worst.z)) worst = { date: x.date, z }
  })
  return {
    count: extreme.length,
    thr: thr.toFixed(1),
    worst: worst ? `${worst.date}（z=${worst.z.toFixed(2)}）` : '—',
    sd: sd.toFixed(2),
  }
})

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
    <div class="mx-auto max-w-6xl">
      <p class="mb-3 text-xs font-semibold tracking-[0.26em] text-sky">EXTREME · 极端事件</p>
      <h1 class="text-3xl font-bold md:text-4xl">极端气候事件日历</h1>
      <p class="mt-3 max-w-2xl text-sm text-dim">按城市统计逐月标准化偏离度（z-score），百分位法（P95/P05）界定极端事件。</p>

      <!-- 城市 + 指标选择 -->
      <section class="mt-10">
        <div class="mb-4 flex flex-wrap gap-2">
          <button
            v-for="c in cities"
            :key="c"
            class="rounded-full border px-4 py-1.5 text-xs transition-all"
            :class="selectedCity === c ? 'border-sky bg-sky/15 text-sky' : 'border-line text-dim hover:text-ice'"
            @click="selectedCity = c"
          >{{ c }}</button>
        </div>
        <div class="flex flex-wrap gap-2">
          <button
            v-for="m in metrics"
            :key="m.key"
            class="rounded-full border px-4 py-1.5 text-xs transition-all"
            :class="selectedMetric === m.key ? 'border-sky bg-sky/15 font-semibold text-sky' : 'border-line text-dim hover:text-ice'"
            @click="selectedMetric = m.key"
          >{{ m.label }} <span class="opacity-70">{{ m.note }}</span></button>
        </div>
      </section>

      <!-- 统计卡片 -->
      <section class="mt-8 grid grid-cols-2 gap-3 md:grid-cols-4">
        <div class="rounded-xl border border-line bg-surface p-4 text-center">
          <div class="text-2xl font-black text-amber">{{ stats.count }}</div>
          <div class="mt-1 text-xs text-dim">极端事件月数（{{ metric.label }}）</div>
        </div>
        <div class="rounded-xl border border-line bg-surface p-4 text-center">
          <div class="text-2xl font-black text-ice">{{ stats.thr }}</div>
          <div class="mt-1 text-xs text-dim">判定阈值（{{ metric.note }}）</div>
        </div>
        <div class="rounded-xl border border-line bg-surface p-4 text-center">
          <div class="text-xl font-black text-sky">{{ stats.worst }}</div>
          <div class="mt-1 text-xs text-dim">偏离最极端月份</div>
        </div>
        <div class="rounded-xl border border-line bg-surface p-4 text-center">
          <div class="text-2xl font-black text-ice">{{ stats.sd }}</div>
          <div class="mt-1 text-xs text-dim">标准差 σ</div>
        </div>
      </section>

      <!-- 热力图 -->
      <section class="mt-8 reveal">
        <div class="rounded-xl border border-line bg-surface p-4">
          <div class="mb-1 flex flex-wrap items-baseline justify-between gap-2">
            <h2 class="text-sm font-bold">{{ metric.label }} · {{ selectedCity }}（2005–2025）</h2>
            <span class="text-xs text-dim">颜色越深偏离越大，红=偏高 / 蓝=偏低</span>
          </div>
          <div style="height: 620px; width: 100%;">
            <VChart :option="heatmapOption" autoresize style="height: 100%; width: 100%;" />
          </div>
        </div>
      </section>

      <p class="mt-6 text-xs text-dim/70">方法：百分位法（P95 高温/多雨、P05 低温/少雨）+ Z-score 标准化，|Z| &gt; 2 视为显著异常。结果基于 8 市逐月再分析数据。</p>
    </div>
  </div>
</template>
