<script setup>
import { onMounted, ref, computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { HeatmapChart, ScatterChart, LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, VisualMapComponent, LegendComponent } from 'echarts/components'
import overview from '../data/overview.json'
import timeseries from '../data/timeseries.json'

use([CanvasRenderer, HeatmapChart, ScatterChart, LineChart, GridComponent, TooltipComponent, VisualMapComponent, LegendComponent])

const cities = overview.stations

// 变量池 (timeseries.json 全部字段)
const varList = [
  { key: 'temp', label: '平均气温' }, { key: 'tmax', label: '最高气温' },
  { key: 'tmin', label: '最低气温' }, { key: 'dew', label: '露点温度' },
  { key: 'precip', label: '降水量' }, { key: 'humidity', label: '相对湿度' },
  { key: 'wind', label: '风速' }, { key: 'pressure', label: '气压' },
  { key: 'radiation', label: '短波辐射' }, { key: 'sunshine', label: '日照时数' },
  { key: 'evap', label: '蒸散发' }, { key: 'cloud', label: '总云量' },
  { key: 'soil', label: '根区土壤湿度' }, { key: 'ndvi', label: 'NDVI' },
  { key: 'spi1', label: 'SPI-1' }, { key: 'spi3', label: 'SPI-3' },
  { key: 'spei1', label: 'SPEI-1' }, { key: 'spei3', label: 'SPEI-3' },
]
const varMap = Object.fromEntries(varList.map((v) => [v.key, v]))
const scatterX = ref('temp')
const scatterY = ref('precip')

// 全部城市月度样本拼接
const allPoints = computed(() => {
  const pts = []
  for (const c of cities) (timeseries[c] || []).forEach((x) => pts.push(x))
  return pts
})

function pearson(xs, ys) {
  const n = xs.length
  if (!n) return 0
  const mx = xs.reduce((a, b) => a + b, 0) / n
  const my = ys.reduce((a, b) => a + b, 0) / n
  let sxy = 0, sxx = 0, syy = 0
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my)
    sxx += (xs[i] - mx) ** 2
    syy += (ys[i] - my) ** 2
  }
  return sxx * syy === 0 ? 0 : sxy / Math.sqrt(sxx * syy)
}
const strength = (r) => {
  const a = Math.abs(r)
  if (a >= 0.8) return '极强相关'
  if (a >= 0.6) return '强相关'
  if (a >= 0.4) return '中等相关'
  if (a >= 0.2) return '弱相关'
  return '极弱相关'
}

// ===== 相关矩阵热力图 =====
const heatmapData = computed(() => {
  const keys = varList.map((v) => v.key)
  const n = keys.length
  const m = Array.from({ length: n }, (_, i) => Array(n).fill(0).map((_, j) => i === j ? 1 : null))
  const cols = keys.map((k) => allPoints.value.map((p) => p[k]))
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const r = pearson(cols[i], cols[j])
      const rv = Number.isFinite(r) ? r : 0
      m[i][j] = rv
      m[j][i] = rv
    }
  }
  const out = []
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      out.push([j, i, m[i][j]])
    }
  }
  return out
})

const heatmapOption = computed(() => ({
  backgroundColor: 'transparent',
  tooltip: {
    position: 'top',
    formatter: (p) => `${varList[p.value[1]].label} × ${varList[p.value[0]].label}<br/>r = <b>${p.value[2].toFixed(3)}</b>`,
  },
  grid: { left: 96, right: 30, top: 16, bottom: 84 },
  xAxis: { type: 'category', data: varList.map((v) => v.label), axisLabel: { color: '#64748b', rotate: 45, fontSize: 10 }, axisLine: { lineStyle: { color: 'rgba(30,41,59,0.15)' } } },
  yAxis: { type: 'category', data: varList.map((v) => v.label), axisLabel: { color: '#64748b', fontSize: 10 }, axisLine: { lineStyle: { color: 'rgba(30,41,59,0.15)' } } },
  visualMap: {
    min: -1, max: 1,
    calculable: true,
    orient: 'horizontal',
    left: 'center',
    bottom: 10,
    itemWidth: 12, itemHeight: 120,
    textStyle: { color: '#64748b' },
    inRange: { color: ['#3b82f6', '#f8fafc', '#ef4444'] },
  },
  series: [
    {
      type: 'heatmap',
      data: heatmapData.value,
      label: {
        show: true,
        formatter: (p) => p.value[2] == null ? '' : p.value[2].toFixed(2),
        fontSize: 9,
        color: (p) => Math.abs(p.value[2]) > 0.6 ? '#ffffff' : Math.abs(p.value[2]) < 0.15 ? '#94a3b8' : '#1e293b',
      },
      emphasis: {
        itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.4)' },
        label: { show: true, fontSize: 12, fontWeight: 'bold' },
      },
    },
  ],
}))

// ===== 散点探索器 =====
const scatterOption = computed(() => {
  const xs = allPoints.value.map((p) => p[scatterX.value])
  const ys = allPoints.value.map((p) => p[scatterY.value])
  const r = pearson(xs, ys)
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length
  const my = ys.reduce((a, b) => a + b, 0) / ys.length
  let sxx = 0, sxy = 0
  for (let i = 0; i < xs.length; i++) {
    sxx += (xs[i] - mx) ** 2
    sxy += (xs[i] - mx) * (ys[i] - my)
  }
  const slope = sxx === 0 ? 0 : sxy / sxx
  const intercept = my - slope * mx
  const xmin = Math.min(...xs), xmax = Math.max(...xs)
  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      formatter: (p) => `${p.value[0].toFixed(1)} , ${p.value[1].toFixed(1)}`,
    },
    grid: { left: 64, right: 30, top: 40, bottom: 48 },
    xAxis: { type: 'value', name: varMap[scatterX.value].label, nameTextStyle: { color: '#64748b' }, axisLabel: { color: '#64748b' }, splitLine: { lineStyle: { color: 'rgba(30,41,59,0.08)' } } },
    yAxis: { type: 'value', name: varMap[scatterY.value].label, nameTextStyle: { color: '#64748b' }, axisLabel: { color: '#64748b' }, splitLine: { lineStyle: { color: 'rgba(30,41,59,0.08)' } } },
    series: [
      {
        type: 'scatter',
        name: '月度样本',
        symbolSize: 5,
        itemStyle: { color: 'rgba(66,165,245,0.55)' },
        data: allPoints.value.map((p) => [p[scatterX.value], p[scatterY.value]]),
      },
      {
        type: 'line',
        name: '回归线',
        showSymbol: false,
        lineStyle: { width: 2, color: '#ff9848' },
        data: [
          [xmin, slope * xmin + intercept],
          [xmax, slope * xmax + intercept],
        ],
      },
    ],
  }
})
const scatterR = computed(() => {
  const xs = allPoints.value.map((p) => p[scatterX.value])
  const ys = allPoints.value.map((p) => p[scatterY.value])
  return pearson(xs, ys)
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
      <p class="mb-3 text-xs font-semibold tracking-[0.26em] text-sky">CORRELATION · 相关性分析</p>
      <h1 class="text-3xl font-bold md:text-4xl">变量相关性探索</h1>
      <p class="mt-3 max-w-2xl text-sm text-dim">18 项气象变量的 Pearson 相关矩阵，任选两变量查看散点关系与相关系数（基于 8 市全部月度样本）。</p>

      <!-- 相关矩阵热力图 -->
      <section class="mt-10 reveal">
        <div class="rounded-xl border border-line bg-surface p-4">
          <h2 class="mb-1 text-sm font-bold">Pearson 相关矩阵（18 × 18）</h2>
          <div style="height: 680px; width: 100%;">
            <VChart :option="heatmapOption" autoresize style="height: 100%; width: 100%;" />
          </div>
        </div>
      </section>

      <!-- 散点探索器 -->
      <section class="mt-12">
        <h2 class="mb-4 text-xl font-bold">散点探索器</h2>
        <div class="mb-4 space-y-3">
          <div class="flex flex-wrap items-center gap-2">
            <span class="w-14 shrink-0 text-xs font-semibold text-dim">X 变量</span>
            <button
              v-for="v in varList"
              :key="'x' + v.key"
              class="rounded-full border px-2.5 py-1 text-[11px] transition-all"
              :class="scatterX === v.key ? 'border-sky bg-sky/15 font-semibold text-sky' : 'border-line text-dim hover:text-ice'"
              @click="scatterX = v.key"
            >{{ v.label }}</button>
          </div>
          <div class="flex flex-wrap items-center gap-2">
            <span class="w-14 shrink-0 text-xs font-semibold text-dim">Y 变量</span>
            <button
              v-for="v in varList"
              :key="'y' + v.key"
              class="rounded-full border px-2.5 py-1 text-[11px] transition-all"
              :class="scatterY === v.key ? 'border-sky bg-sky/15 font-semibold text-sky' : 'border-line text-dim hover:text-ice'"
              @click="scatterY = v.key"
            >{{ v.label }}</button>
          </div>
        </div>

        <div class="reveal rounded-xl border border-line bg-surface p-4">
          <div class="mb-2 flex flex-wrap items-baseline gap-4">
            <h3 class="text-sm font-bold">{{ varMap[scatterX].label }} × {{ varMap[scatterY].label }}</h3>
            <span class="text-xs text-dim">样本量 {{ allPoints.length }}</span>
            <span class="rounded-full bg-sky/15 px-3 py-0.5 text-xs font-semibold text-sky">
              r = {{ scatterR.toFixed(3) }} · {{ strength(scatterR) }}
            </span>
          </div>
          <div style="height: 420px; width: 100%;">
            <VChart :option="scatterOption" autoresize style="height: 100%; width: 100%;" />
          </div>
        </div>
      </section>
    </div>
  </div>
</template>
