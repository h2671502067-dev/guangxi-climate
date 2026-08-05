<script setup>
import { onMounted, ref, computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { RadarChart, ScatterChart, LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, RadarComponent } from 'echarts/components'
import clustering from '../data/clustering.json'

use([CanvasRenderer, RadarChart, ScatterChart, LineChart, GridComponent, TooltipComponent, LegendComponent, RadarComponent])

const months = clustering.months
const clusterIds = [...new Set(months.map((m) => m.cluster))].sort((a, b) => a - b)
const clusterColors = ['#ff9848', '#42a5f5', '#5ccfe6', '#a78bfa', '#4ade80', '#fbbf24', '#ef4444', '#64748b']

const features = [
  { key: 'temp', name: '气温(°C)' },
  { key: 'precip', name: '降水(mm)' },
  { key: 'humidity', name: '湿度(%)' },
  { key: 'wind', name: '风速(km/h)' },
  { key: 'sunshine', name: '日照(h)' },
]

// 月份归属表
const groups = computed(() =>
  clusterIds.map((cid) => ({
    cid,
    months: months.filter((m) => m.cluster === cid).map((m) => m.month).sort((a, b) => a - b),
  }))
)

// 雷达图 (簇质心)
const radarOption = computed(() => {
  const indicator = features.map((f) => {
    const vals = months.map((m) => m[f.key])
    return { name: f.name, min: Math.min(...vals), max: Math.max(...vals) }
  })
  return {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'item' },
    legend: { data: clusterIds.map((c) => `簇${c}`), textStyle: { color: '#64748b' }, top: 0 },
    radar: {
      indicator,
      radius: '60%',
      center: ['50%', '56%'],
      axisName: { color: '#64748b' },
      splitLine: { lineStyle: { color: 'rgba(30,41,59,0.15)' } },
      splitArea: { areaStyle: { color: ['rgba(66,165,245,0.03)', 'rgba(66,165,245,0.07)'] } },
      axisLine: { lineStyle: { color: 'rgba(30,41,59,0.15)' } },
    },
    series: [
      {
        type: 'radar',
        data: clusterIds.map((cid) => {
          const mem = months.filter((m) => m.cluster === cid)
          const vals = features.map((f) => mem.reduce((a, m) => a + m[f.key], 0) / mem.length)
          return {
            name: `簇${cid}`,
            value: vals,
            lineStyle: { width: 1.6 },
            itemStyle: { color: clusterColors[cid] },
            areaStyle: { opacity: 0.12 },
            label: {
              show: true,
              position: 'top',
              distance: 4,
              formatter: (p) => vals[p.dataIndex].toFixed(1),
              fontSize: 10,
              color: clusterColors[cid],
              textBorderColor: '#ffffff',
              textBorderWidth: 2,
            },
          }
        }),
      },
    ],
  }
})

// PCA 散点
const pcaOption = computed(() => ({
  backgroundColor: 'transparent',
  tooltip: { trigger: 'item', formatter: (p) => `${p.seriesName}<br/>${p.data[2]}月` },
  legend: { data: clusterIds.map((c) => `簇${c}`), textStyle: { color: '#64748b' }, top: 0 },
  grid: { left: 56, right: 30, top: 40, bottom: 48 },
  xAxis: { type: 'value', name: `PC1 (${clustering.pca_variance[0]}%)`, nameTextStyle: { color: '#64748b' }, axisLabel: { color: '#64748b' }, splitLine: { lineStyle: { color: 'rgba(30,41,59,0.08)' } } },
  yAxis: { type: 'value', name: `PC2 (${clustering.pca_variance[1]}%)`, nameTextStyle: { color: '#64748b' }, axisLabel: { color: '#64748b' }, splitLine: { lineStyle: { color: 'rgba(30,41,59,0.08)' } } },
  series: clusterIds.map((cid) => ({
    type: 'scatter',
    name: `簇${cid}`,
    symbolSize: 16,
    itemStyle: { color: clusterColors[cid] },
    label: { show: true, formatter: (p) => `${p.data[2]}月`, position: 'top', fontSize: 10, color: '#64748b' },
    data: months.filter((m) => m.cluster === cid).map((m) => [m.pc1, m.pc2, m.month]),
  })),
}))

// 肘部法则 + 轮廓系数
const kList = Object.keys(clustering.inertias).map(Number).sort((a, b) => a - b)
const elbowOption = computed(() => ({
  backgroundColor: 'transparent',
  tooltip: { trigger: 'axis' },
  legend: { data: ['WCSS', '轮廓系数'], textStyle: { color: '#64748b' }, top: 0 },
  grid: { left: 56, right: 56, top: 40, bottom: 44 },
  xAxis: { type: 'category', data: kList.map(String), name: '聚类数 K', nameTextStyle: { color: '#64748b' }, axisLabel: { color: '#64748b' }, axisLine: { lineStyle: { color: 'rgba(30,41,59,0.15)' } } },
  yAxis: [
    { type: 'value', name: 'WCSS', nameTextStyle: { color: '#64748b' }, axisLabel: { color: '#64748b' }, splitLine: { lineStyle: { color: 'rgba(30,41,59,0.08)' } } },
    { type: 'value', name: '轮廓系数', nameTextStyle: { color: '#64748b' }, axisLabel: { color: '#64748b' }, splitLine: { show: false } },
  ],
  series: [
    {
      name: 'WCSS', type: 'line', symbolSize: 7, itemStyle: { color: '#ff9848' }, lineStyle: { width: 2, color: '#ff9848' },
      data: kList.map((k) => clustering.inertias[k]),
      markLine: { silent: true, lineStyle: { type: 'dashed', color: 'rgba(30,41,59,0.4)' }, data: [{ xAxis: String(clustering.optimal_k) }], label: { formatter: `最优 K=${clustering.optimal_k}`, color: '#64748b', fontSize: 10 } },
    },
    { name: '轮廓系数', type: 'line', yAxisIndex: 1, symbolSize: 7, itemStyle: { color: '#42a5f5' }, lineStyle: { width: 2, color: '#42a5f5' }, data: kList.map((k) => clustering.silhouettes[k]) },
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
      <p class="mb-3 text-xs font-semibold tracking-[0.26em] text-sky">CLUSTERING · 气候区划</p>
      <h1 class="text-3xl font-bold md:text-4xl">月份气候聚类分析</h1>
      <p class="mt-3 max-w-2xl text-sm text-dim">K-Means 聚类将 12 个月份按 5 项气象特征划分气候类型，PCA 降维可视化。</p>

      <!-- 信息卡片 -->
      <section class="mt-10 grid grid-cols-2 gap-3 md:grid-cols-4">
        <div class="rounded-xl border border-line bg-surface p-4 text-center">
          <div class="text-2xl font-black text-amber">{{ clustering.optimal_k }}</div>
          <div class="mt-1 text-xs text-dim">最优聚类数 K（肘部法则）</div>
        </div>
        <div class="rounded-xl border border-line bg-surface p-4 text-center">
          <div class="text-2xl font-black text-sky">{{ clustering.silhouettes[String(clustering.optimal_k)] }}</div>
          <div class="mt-1 text-xs text-dim">轮廓系数（K={{ clustering.optimal_k }}）</div>
        </div>
        <div class="rounded-xl border border-line bg-surface p-4 text-center">
          <div class="text-2xl font-black text-ice">{{ (clustering.pca_variance[0] + clustering.pca_variance[1]).toFixed(1) }}%</div>
          <div class="mt-1 text-xs text-dim">PC1 + PC2 累计解释方差</div>
        </div>
        <div class="rounded-xl border border-line bg-surface p-4 text-center">
          <div class="text-2xl font-black text-ice">{{ months.length }}</div>
          <div class="mt-1 text-xs text-dim">聚类样本（12 个月份）</div>
        </div>
      </section>

      <!-- 月份归属 -->
      <section class="mt-10">
        <h2 class="mb-4 text-xl font-bold">各簇月份组成</h2>
        <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <div
            v-for="g in groups"
            :key="g.cid"
            class="reveal rounded-xl border border-line bg-surface p-5"
          >
            <div class="flex items-center gap-2">
              <span class="h-3 w-3 rounded-full" :style="{ background: clusterColors[g.cid] }"></span>
              <h3 class="text-sm font-bold">簇 {{ g.cid }}</h3>
            </div>
            <div class="mt-3 flex flex-wrap gap-1.5">
              <span
                v-for="m in g.months"
                :key="m"
                class="rounded-md px-2.5 py-1 text-xs"
                :style="{ background: clusterColors[g.cid] + '1a', color: clusterColors[g.cid] }"
              >{{ m }}月</span>
            </div>
          </div>
        </div>
      </section>

      <!-- 雷达 + PCA -->
      <section class="mt-12 grid gap-6 lg:grid-cols-2">
        <div class="reveal rounded-xl border border-line bg-surface p-4">
          <h2 class="mb-1 text-sm font-bold">簇质心特征画像（雷达图）</h2>
          <div style="height: 400px; width: 100%;">
            <VChart :option="radarOption" autoresize style="height: 100%; width: 100%;" />
          </div>
        </div>
        <div class="reveal rounded-xl border border-line bg-surface p-4">
          <h2 class="mb-1 text-sm font-bold">PCA 降维聚类散点</h2>
          <div style="height: 400px; width: 100%;">
            <VChart :option="pcaOption" autoresize style="height: 100%; width: 100%;" />
          </div>
        </div>
      </section>

      <!-- 选K过程 -->
      <section class="mt-12 reveal">
        <div class="rounded-xl border border-line bg-surface p-4">
          <h2 class="mb-1 text-sm font-bold">最优 K 选择过程</h2>
          <div style="height: 320px; width: 100%;">
            <VChart :option="elbowOption" autoresize style="height: 100%; width: 100%;" />
          </div>
        </div>
      </section>

      <p class="mt-6 text-xs text-dim/70">方法：K-Means（肘部法则 + 轮廓系数自动选 K）+ PCA 降维。样本量仅 12 个月份，聚类结果仅供参考，与春夏秋冬季节高度吻合。</p>
    </div>
  </div>
</template>
