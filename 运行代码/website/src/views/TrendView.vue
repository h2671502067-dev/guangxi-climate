<script setup>
import { onMounted, ref, computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import overview from '../data/overview.json'

use([CanvasRenderer, BarChart, GridComponent, TooltipComponent])

const trends = overview.trends

const isUp = (t) => t.trend.includes('上升')
const isDown = (t) => t.trend.includes('下降')
const sig = (t) => !t.trend.includes('不')
const arrow = (t) => (isUp(t) ? '↑' : isDown(t) ? '↓' : '→')
const colorOf = (t) => {
  if (isUp(t) && sig(t)) return '#ef4444'
  if (isDown(t) && sig(t)) return '#3b82f6'
  return '#94a3b8'
}

const nSig = computed(() => trends.filter((t) => sig(t)).length)

const barOption = computed(() => ({
  backgroundColor: 'transparent',
  tooltip: {
    trigger: 'axis',
    axisPointer: { type: 'shadow' },
    formatter: (params) => {
      const p = params[0]
      const t = trends[p.dataIndex]
      return `${t.col}<br/>Z 值：<b>${t.z}</b><br/>p 值：${t.p}<br/>结论：${t.trend}`
    },
  },
  grid: { left: 130, right: 40, top: 20, bottom: 40 },
  xAxis: {
    type: 'value',
    name: 'Mann-Kendall Z 值',
    nameTextStyle: { color: '#64748b' },
    axisLabel: { color: '#64748b' },
    splitLine: { lineStyle: { color: 'rgba(30,41,59,0.08)' } },
  },
  yAxis: {
    type: 'category',
    data: trends.map((t) => t.col).reverse(),
    axisLabel: { color: '#64748b' },
    axisLine: { lineStyle: { color: 'rgba(30,41,59,0.15)' } },
  },
  series: [
    {
      type: 'bar',
      barWidth: 18,
      data: [...trends].reverse().map((t) => ({
        value: t.z,
        itemStyle: { color: colorOf(t), borderRadius: [0, 4, 4, 0] },
      })),
      label: {
        show: true,
        position: 'right',
        formatter: (p) => `${arrow([...trends].reverse()[p.dataIndex])} Z=${p.value}`,
        color: '#1e293b',
        fontSize: 11,
      },
      markLine: {
        silent: true,
        lineStyle: { type: 'dashed', color: 'rgba(30,41,59,0.3)' },
        data: [{ xAxis: 1.96 }, { xAxis: -1.96 }],
        label: { show: false },
      },
    },
  ],
}))

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
      <p class="mb-3 text-xs font-semibold tracking-[0.26em] text-sky">TREND · 趋势检验</p>
      <h1 class="text-3xl font-bold md:text-4xl">Mann-Kendall 气候趋势检验</h1>
      <p class="mt-3 max-w-2xl text-sm text-dim">非参数秩检验判断各气象要素的长期变化趋势，|Z| &gt; 1.96 表示 p &lt; 0.05 显著。</p>

      <!-- 摘要横幅 -->
      <section class="mt-10 flex flex-wrap items-center gap-6 rounded-2xl border border-sky/30 bg-gradient-to-r from-surface to-ink p-6">
        <div class="text-4xl font-black text-amber">{{ nSig }} / {{ trends.length }}</div>
        <div>
          <p class="text-sm font-bold">个要素呈显著趋势变化</p>
          <p class="mt-1 text-xs text-dim">其中平均气温与总降水量均为极显著上升（Z=2.99, p&lt;0.01），广西气候呈明显暖湿化。</p>
        </div>
      </section>

      <!-- 变量卡片 -->
      <section class="mt-10 grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <article
          v-for="t in trends"
          :key="t.col"
          class="reveal rounded-xl border border-line bg-surface p-5"
        >
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-ice">{{ t.col }}</h3>
            <span
              class="rounded-full px-3 py-1 text-xs font-bold"
              :style="{ background: colorOf(t) + '1a', color: colorOf(t) }"
            >{{ arrow(t) }} {{ t.trend }}</span>
          </div>
          <div class="mt-4 flex gap-6 text-xs">
            <div>
              <div class="text-xl font-black" :style="{ color: colorOf(t) }">{{ t.z }}</div>
              <div class="mt-1 text-dim">Z 值</div>
            </div>
            <div>
              <div class="text-xl font-black text-ice">{{ t.p }}</div>
              <div class="mt-1 text-dim">p 值</div>
            </div>
          </div>
        </article>
      </section>

      <!-- Z 值条形图 -->
      <section class="mt-12 reveal">
        <div class="rounded-xl border border-line bg-surface p-4">
          <h2 class="mb-1 text-sm font-bold">各要素趋势强度（Z 值）</h2>
          <div style="height: 360px; width: 100%;">
            <VChart :option="barOption" autoresize style="height: 100%; width: 100%;" />
          </div>
        </div>
      </section>

      <p class="mt-6 text-xs text-dim/70">方法：Mann-Kendall 秩检验 + Sen 斜率估计，含 ties 修正。虚线标记 |Z|=1.96（95% 置信水平）。</p>
    </div>
  </div>
</template>
