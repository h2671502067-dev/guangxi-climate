<script setup>
import { onMounted, ref, computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, DataZoomComponent } from 'echarts/components'
import overview from '../data/overview.json'
import timeseries from '../data/timeseries.json'

use([CanvasRenderer, LineChart, BarChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent])

const cities = overview.stations
const selectedCity = ref('南宁')

const indexKeys = [
  { key: 'spi1', label: 'SPI-1', sub: '1个月标准化降水指数' },
  { key: 'spi3', label: 'SPI-3', sub: '3个月标准化降水指数' },
  { key: 'spei1', label: 'SPEI-1', sub: '1个月标准化降水蒸散指数' },
  { key: 'spei3', label: 'SPEI-3', sub: '3个月标准化降水蒸散指数' },
]
const selectedIndex = ref('spi3')
const label = computed(() => indexKeys.find((i) => i.key === selectedIndex.value).label)

// USDM 干旱等级分片
const pieces = [
  { lt: -2, label: 'D4 异常干旱', color: '#cc0000' },
  { gte: -2, lt: -1.6, label: 'D3 极端干旱', color: '#ff6600' },
  { gte: -1.6, lt: -1.3, label: 'D2 严重干旱', color: '#ff9966' },
  { gte: -1.3, lt: -0.8, label: 'D1 中度干旱', color: '#ffcc99' },
  { gte: -0.8, lt: -0.5, label: 'D0 轻度干旱', color: '#ffff00' },
  { gte: -0.5, lt: 1, label: '接近正常', color: '#8bc34a' },
  { gte: 1, lt: 1.5, label: '中度湿润', color: '#abd9e9' },
  { gte: 1.5, lt: 2, label: '重度湿润', color: '#2c7bb6' },
  { gte: 2, label: '极度湿润', color: '#0b3d91' },
]

function levelColor(v) {
  if (v < -2) return '#cc0000'
  if (v < -1.6) return '#ff6600'
  if (v < -1.3) return '#ff9966'
  if (v < -0.8) return '#ffcc99'
  if (v < -0.5) return '#ffff00'
  if (v < 0.5) return '#8bc34a'
  if (v < 1) return '#abd9e9'
  if (v < 1.5) return '#2c7bb6'
  return '#0b3d91'
}
function levelLabel(v) {
  const p = pieces.find((x) => v < -2 ? x.lt === -2 && x.gte === undefined : (x.gte !== undefined && v >= x.gte && (x.lt === undefined || v < x.lt)))
  return p ? p.label : '接近正常'
}

const raw = computed(() => timeseries[selectedCity.value] || [])

// 统计卡片
const stats = computed(() => {
  const d = raw.value
  const vals = d.map((x) => x[selectedIndex.value])
  let min = Infinity, minDate = '', max = -Infinity, maxDate = ''
  vals.forEach((v, i) => {
    if (v == null) return
    if (v < min) { min = v; minDate = d[i].date }
    if (v > max) { max = v; maxDate = d[i].date }
  })
  const c = (lo, hi) => vals.filter((v) => v != null && (lo == null || v >= lo) && (hi == null || v < hi)).length
  return {
    dry: c(-0.5, null), d0: c(-0.8, -0.5), d1: c(-1.3, -0.8),
    d2: c(-1.6, -1.3), d3: c(-2, -1.6), d4: c(null, -2),
    wet: c(0.5, null), min, minDate, max, maxDate,
  }
})

const chartOption = computed(() => {
  const d = raw.value
  const dates = d.map((x) => x.date)
  const values = d.map((x) => x[selectedIndex.value])
  // 色带数据：每格单独一个颜色
  const ribbon = values.map((v) => ({
    value: 1,
    itemStyle: { color: levelColor(v) },
    tooltipExtra: {
      value: v,
      label: levelLabel(v),
    },
  }))
  return {
    backgroundColor: 'transparent',
    title: {
      text: `${label.value} · ${selectedCity.value}`,
      left: 56,
      top: 8,
      textStyle: { color: '#1e293b', fontSize: 13, fontWeight: 'bold' },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(30,41,59,0.92)',
      borderWidth: 0,
      textStyle: { color: '#f8fafc', fontSize: 12 },
      axisPointer: { type: 'line', lineStyle: { color: 'rgba(66,165,245,0.4)' } },
      formatter: (params) => {
        const p = params[0]
        if (p.seriesIndex === 1) {
          const extra = p.data.tooltipExtra
          return `${p.axisValue}<br/>${label.value}：<b>${extra.value}</b><br/>${extra.label}`
        }
        return `${p.axisValue}<br/>${label.value}：<b>${p.value}</b>`
      },
    },
    // 自定义 legend 改为模板渲染 (更稳定, 不受 ECharts 布局约束)
    grid: [
      { left: 56, right: 30, top: 40, bottom: 120, height: '58%' },
      { left: 56, right: 30, top: 228, bottom: 90, height: '10%' },
    ],
    xAxis: [
      {
        type: 'category',
        data: dates,
        gridIndex: 0,
        boundaryGap: false,
        axisLabel: { show: false },
        axisLine: { lineStyle: { color: 'rgba(30,41,59,0.15)' } },
        axisTick: { show: false },
      },
      {
        type: 'category',
        data: dates,
        gridIndex: 1,
        boundaryGap: false,
        axisLabel: { show: false },
        axisLine: { show: false },
        axisTick: { show: false },
      },
    ],
    yAxis: [
      {
        type: 'value',
        gridIndex: 0,
        name: label.value,
        nameTextStyle: { color: '#64748b', fontSize: 11 },
        axisLabel: { color: '#64748b' },
        splitLine: { lineStyle: { color: 'rgba(30,41,59,0.08)' } },
      },
      {
        type: 'value',
        gridIndex: 1,
        min: 0,
        max: 1,
        show: false,
      },
    ],
    dataZoom: [
      { type: 'inside', xAxisIndex: [0, 1], start: 0, end: 100, zoomLock: false },
      {
        type: 'slider',
        xAxisIndex: [0, 1],
        height: 18,
        bottom: 60,
        borderColor: 'rgba(30,41,59,0.1)',
        backgroundColor: 'rgba(255,255,255,0.6)',
        fillerColor: 'rgba(66,165,245,0.18)',
        handleStyle: { color: '#42a5f5' },
        textStyle: { color: '#64748b' },
      },
    ],
    series: [
      {
        name: label.value,
        type: 'line',
        gridIndex: 0,
        showSymbol: false,
        smooth: true,
        connectNulls: true,
        lineStyle: { width: 2, color: '#42a5f5' },
        itemStyle: { color: '#42a5f5' },
        areaStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(66,165,245,0.25)' },
              { offset: 1, color: 'rgba(66,165,245,0.02)' },
            ],
          },
        },
        markLine: {
          silent: true,
          symbol: 'none',
          lineStyle: { type: 'dashed', color: 'rgba(30,41,59,0.3)' },
          data: [{ yAxis: 0 }],
        },
        data: values,
      },
      {
        name: '干旱等级',
        type: 'bar',
        gridIndex: 1,
        barWidth: '86%',
        barGap: '-100%',
        itemStyle: { borderWidth: 0 },
        data: ribbon,
      },
    ],
  }
})

// NDVI × 降水联动
const vegOption = computed(() => {
  const d = raw.value
  return {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis' },
    legend: { data: ['NDVI 植被指数', '降水量'], textStyle: { color: '#64748b' }, top: 0 },
    grid: { left: 56, right: 56, top: 44, bottom: 60 },
    xAxis: { type: 'category', data: d.map((x) => x.date), axisLabel: { color: '#64748b', interval: 23 }, axisLine: { lineStyle: { color: 'rgba(30,41,59,0.15)' } } },
    yAxis: [
      { type: 'value', name: 'NDVI', nameTextStyle: { color: '#64748b' }, axisLabel: { color: '#64748b' }, splitLine: { lineStyle: { color: 'rgba(30,41,59,0.08)' } } },
      { type: 'value', name: 'mm/月', nameTextStyle: { color: '#64748b' }, axisLabel: { color: '#64748b' }, splitLine: { show: false } },
    ],
    dataZoom: [
      { type: 'inside', start: 0, end: 100 },
      { type: 'slider', height: 20, bottom: 6 },
    ],
    series: [
      { name: 'NDVI 植被指数', type: 'line', showSymbol: false, smooth: true, lineStyle: { width: 2, color: '#4ade80' }, data: d.map((x) => x.ndvi) },
      { name: '降水量', type: 'bar', yAxisIndex: 1, barMaxWidth: 3, itemStyle: { color: 'rgba(66,165,245,0.5)' }, data: d.map((x) => x.precip) },
    ],
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
      <p class="mb-3 text-xs font-semibold tracking-[0.26em] text-sky">DROUGHT · 干旱监测</p>
      <h1 class="text-3xl font-bold md:text-4xl">干旱指数与植被监测</h1>
      <p class="mt-3 max-w-2xl text-sm text-dim">SPI / SPEI 干旱指数时序 + 干旱等级色带（USDM 标准 D0–D4），联动 NDVI 植被健康。</p>

      <!-- 城市 + 指数选择 -->
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
            v-for="i in indexKeys"
            :key="i.key"
            class="rounded-full border px-4 py-1.5 text-xs transition-all"
            :class="selectedIndex === i.key ? 'border-sky bg-sky/15 font-semibold text-sky' : 'border-line text-dim hover:text-ice'"
            @click="selectedIndex = i.key"
          >{{ i.label }}</button>
        </div>
      </section>

      <!-- 统计卡片 -->
      <section class="mt-8 grid grid-cols-2 gap-3 md:grid-cols-4">
        <div class="rounded-xl border border-line bg-surface p-4 text-center">
          <div class="text-2xl font-black text-amber">{{ stats.dry }}</div>
          <div class="mt-1 text-xs text-dim">轻度及以上干旱月数（D0+）</div>
        </div>
        <div class="rounded-xl border border-line bg-surface p-4 text-center">
          <div class="text-2xl font-black text-red-600">{{ stats.d2 + stats.d3 + stats.d4 }}</div>
          <div class="mt-1 text-xs text-dim">严重干旱月数（D2+）</div>
        </div>
        <div class="rounded-xl border border-line bg-surface p-4 text-center">
          <div class="text-2xl font-black text-sky">{{ stats.wet }}</div>
          <div class="mt-1 text-xs text-dim">湿润月数（&gt;0.5）</div>
        </div>
        <div class="rounded-xl border border-line bg-surface p-4 text-center">
          <div class="text-2xl font-black text-ice">{{ stats.min }}<span class="text-xs text-dim"> · {{ stats.minDate }}</span></div>
          <div class="mt-1 text-xs text-dim">历史最干（{{ label }} 最低）</div>
        </div>
      </section>

      <!-- 指数时序图 -->
      <section class="mt-8 reveal">
        <div class="rounded-xl border border-line bg-surface p-4">
          <div class="mb-2 flex flex-wrap items-baseline justify-between gap-2">
            <h2 class="text-sm font-bold">{{ label }} 时序 · {{ selectedCity }}</h2>
            <span class="text-xs text-dim">USDM 分级标准</span>
          </div>
          <!-- 外部渲染图例 (稳定显示) -->
          <div class="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-line pb-2">
            <span
              v-for="p in pieces"
              :key="p.label"
              class="flex items-center gap-1.5 text-[11px]"
            >
              <span
                class="inline-block h-3 w-3 rounded-sm"
                :style="{ background: p.color }"
              ></span>
              <span class="text-dim">{{ p.label }}</span>
            </span>
          </div>
          <div style="height: 430px; width: 100%;">
            <VChart :option="chartOption" autoresize style="height: 100%; width: 100%;" />
          </div>
        </div>
      </section>

      <!-- NDVI 联动 -->
      <section class="mt-8 reveal">
        <div class="rounded-xl border border-line bg-surface p-4">
          <h2 class="mb-1 text-sm font-bold">NDVI 植被指数 × 降水量 · {{ selectedCity }}</h2>
          <div style="height: 320px; width: 100%;">
            <VChart :option="vegOption" autoresize style="height: 100%; width: 100%;" />
          </div>
        </div>
      </section>

      <p class="mt-6 text-xs text-dim/70">干旱等级采用美国干旱监测（USDM）标准：D0 轻度 / D1 中度 / D2 严重 / D3 极端 / D4 异常。SPI/SPEI 来自 ERA5 再分析数据，仅供研究参考。</p>
    </div>
  </div>
</template>
