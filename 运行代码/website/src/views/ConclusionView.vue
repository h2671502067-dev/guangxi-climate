<script setup>
import { onMounted } from 'vue'
import models from '../data/models.json'
import findings from '../data/findings.json'
import overview from '../data/overview.json'

const best = findings.best
const transferGain = findings.transfer_gain

const summaryRows = [
  { key: '样本规模', val: `${overview.n_samples} 个月度观测` },
  { key: '特征维度', val: `${overview.n_features} 维` },
  { key: '时间跨度', val: `${overview.date_min} ~ ${overview.date_max}` },
  { key: '年均气温', val: `${overview.avg_temp}°C` },
  { key: '月均降水', val: `${overview.avg_precip} mm` },
  { key: '降水变异系数', val: `${overview.precip_cv}%` },
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
    <div class="mx-auto max-w-5xl">
      <p class="mb-3 text-xs font-semibold tracking-[0.26em] text-sky">CONCLUSION · 结论</p>
      <h1 class="text-3xl font-bold md:text-4xl">核心发现与模型表现</h1>

      <!-- 最佳模型横幅 -->
      <section class="mt-10 overflow-hidden rounded-2xl border border-sky/30 bg-gradient-to-r from-surface to-ink p-8">
        <div class="flex flex-wrap items-center justify-between gap-6">
          <div>
            <p class="text-xs tracking-widest text-dim">全局最优预测</p>
            <h2 class="mt-2 text-3xl font-black text-sky">{{ best.city }} · {{ best.model }}</h2>
            <p class="mt-1 text-sm text-dim">{{ best.strategy }}</p>
          </div>
          <div class="flex gap-10">
            <div class="text-center">
              <div class="text-4xl font-black text-amber">{{ best.rmse }}</div>
              <div class="mt-1 text-xs text-dim">RMSE (°C)</div>
            </div>
            <div class="text-center">
              <div class="text-4xl font-black text-sky">{{ best.r2 }}</div>
              <div class="mt-1 text-xs text-dim">R²</div>
            </div>
          </div>
        </div>
      </section>

      <!-- 关键发现 -->
      <section class="mt-14">
        <h2 class="mb-6 text-xl font-bold">四大关键发现</h2>
        <div class="grid gap-4 md:grid-cols-2">
          <article v-for="f in findings.core" :key="f.title" class="reveal border-l-2 border-amber bg-surface p-6">
            <h3 class="font-bold text-sky">{{ f.title }}</h3>
            <p class="mt-2 text-sm text-dim">{{ f.desc }}</p>
          </article>
        </div>
      </section>

      <!-- 迁移学习收益 -->
      <section v-if="transferGain.length" class="mt-14">
        <h2 class="mb-6 text-xl font-bold">TCN 迁移学习收益城市</h2>
        <div class="overflow-x-auto rounded-xl border border-line">
          <table class="w-full text-sm">
            <thead>
              <tr class="bg-surface text-left text-dim">
                <th class="px-6 py-3">城市</th>
                <th class="px-6 py-3">RMSE 改善</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="g in transferGain" :key="g.city" class="border-t border-line">
                <td class="px-6 py-3 text-ice">{{ g.city }}</td>
                <td class="px-6 py-3 font-semibold text-amber">+{{ g.gain }}%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <!-- 模型对比表 -->
      <section class="mt-14">
        <h2 class="mb-6 text-xl font-bold">各市最优模型（实测）</h2>
        <div class="overflow-x-auto rounded-xl border border-line">
          <table class="w-full text-sm">
            <thead>
              <tr class="bg-surface text-left text-dim">
                <th class="px-6 py-3">城市</th>
                <th class="px-6 py-3">最优模型</th>
                <th class="px-6 py-3">策略</th>
                <th class="px-6 py-3">RMSE</th>
                <th class="px-6 py-3">R²</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="row in models.per_city"
                :key="row.city"
                class="border-t border-line transition-colors hover:bg-surface"
                :class="row.city === best.city ? 'bg-sky/5' : ''"
              >
                <td class="px-6 py-3">{{ row.city }}</td>
                <td class="px-6 py-3 font-semibold text-sky">{{ row.model }}</td>
                <td class="px-6 py-3 text-dim">{{ row.strategy }}</td>
                <td class="px-6 py-3 font-mono font-bold" :class="row.city === best.city ? 'text-amber' : 'text-ice'">{{ row.rmse.toFixed(3) }}</td>
                <td class="px-6 py-3 font-mono text-dim">{{ row.r2.toFixed(4) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="mt-4 flex flex-wrap gap-3 text-xs text-dim">
          <span v-for="b in models.benchmarks" :key="b.name" class="rounded-full border border-line px-4 py-1.5">
            {{ b.name }} · RMSE {{ b.rmse }} · R² {{ b.r2 }}
          </span>
        </div>
      </section>

      <!-- 数据摘要 -->
      <section class="mt-14">
        <h2 class="mb-6 text-xl font-bold">数据摘要</h2>
        <div class="grid gap-3 md:grid-cols-3">
          <div v-for="s in summaryRows" :key="s.key" class="border border-line bg-surface px-5 py-4">
            <div class="text-xs text-dim">{{ s.key }}</div>
            <div class="mt-1 font-bold text-ice">{{ s.val }}</div>
          </div>
        </div>
      </section>

      <!-- 免责声明 -->
      <section class="mt-14 rounded-xl border border-line bg-ink-alt p-6">
        <h2 class="mb-3 text-sm font-bold text-dim">免责声明</h2>
        <p class="text-xs leading-relaxed text-dim/70">
          本报告基于 NASA POWER、NCEP/NCAR R1 及 MODIS 遥感再分析数据生成，数据可能存在一定偏差。分析结果仅供学术研究和参考，不应作为正式气象预报或决策依据。如需精确气象数据，请参考中国气象局官方发布。
        </p>
      </section>
    </div>
  </div>
</template>
