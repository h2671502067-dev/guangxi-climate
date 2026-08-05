<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import AiAssistant from './components/AiAssistant.vue'

const route = useRoute()
const navOpen = ref(false)
const scrolled = ref(false)

function onScroll() {
  scrolled.value = window.scrollY > 50
}

onMounted(() => window.addEventListener('scroll', onScroll, { passive: true }))
onUnmounted(() => window.removeEventListener('scroll', onScroll))

const links = [
  { to: '/', label: '首页' },
  { to: '/dashboard', label: '数据分析' },
  { to: '/drought', label: '干旱监测' },
  { to: '/trend', label: '气候趋势' },
  { to: '/correlation', label: '相关性' },
  { to: '/extreme', label: '极端事件' },
  { to: '/clustering', label: '城市聚类' },
  { to: '/methods', label: '方法论' },
  { to: '/conclusion', label: '结论' },
]
</script>

<template>
  <div class="min-h-screen flex flex-col bg-ink">
    <header class="sticky top-0 z-40 transition-all duration-300" :class="scrolled || route.path !== '/' ? 'border-b border-line bg-white/80 shadow-sm backdrop-blur-lg' : 'border-b border-transparent bg-transparent'">
      <div class="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <router-link to="/" class="flex items-center gap-3">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" class="h-10 w-10">
            <defs>
              <linearGradient id="logo-fg" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#42a5f5" />
                <stop offset="100%" stop-color="#1d4ed8" />
              </linearGradient>
            </defs>
            <!-- 太阳核心（白色填充+蓝色描边） -->
            <circle cx="10.5" cy="10" r="3.2" fill="#ffffff" stroke="url(#logo-fg)" stroke-width="1.6"></circle>
            <!-- 太阳光线（蓝色描边） -->
            <g class="logo-sun-rays" stroke="url(#logo-fg)" stroke-width="1.6" stroke-linecap="round">
              <line x1="10.5" y1="3.5" x2="10.5" y2="5.2"></line>
              <line x1="10.5" y1="14.8" x2="10.5" y2="16.5"></line>
              <line x1="4.5" y1="10" x2="6.2" y2="10"></line>
              <line x1="14.8" y1="10" x2="16.5" y2="10"></line>
              <line x1="6.5" y1="6.5" x2="7.7" y2="7.7"></line>
              <line x1="13.3" y1="12.3" x2="14.5" y2="13.5"></line>
              <line x1="6.5" y1="13.5" x2="7.7" y2="12.3"></line>
              <line x1="13.3" y1="7.7" x2="14.5" y2="6.5"></line>
            </g>
            <!-- 云朵（白色填充+蓝色描边） -->
            <path class="logo-cloud" d="M24 21h-12a4.2 4.2 0 0 1-.1-8.4 5.8 5.8 0 0 1 11.3 2.1 3.8 3.8 0 0 1 1.2 6.3H24z" fill="#ffffff" stroke="url(#logo-fg)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"></path>
          </svg>
          <span class="hidden h-6 w-px bg-gradient-to-b from-transparent via-sky to-transparent sm:block"></span>
          <span class="text-xl font-bold tracking-widest text-ice">广西气象数据建模分析</span>
        </router-link>
        <nav class="hidden items-center gap-1 md:flex">
          <router-link
            v-for="l in links"
            :key="l.to"
            :to="l.to"
            class="nav-cloud relative px-4 py-1.5 text-sm whitespace-nowrap transition-all duration-200"
            :class="route.path === l.to
              ? 'nav-active font-semibold'
              : 'hover:text-sky'"
          >{{ l.label }}</router-link>
        </nav>
        <button class="md:hidden text-ice" @click="navOpen = !navOpen">☰</button>
      </div>
      <nav v-if="navOpen" class="border-t border-line px-6 py-3 md:hidden">
        <router-link
          v-for="l in links"
          :key="l.to"
          :to="l.to"
          class="block py-2 text-sm"
          :class="route.path === l.to ? 'text-sky' : 'text-dim'"
          @click="navOpen = false"
        >{{ l.label }}</router-link>
      </nav>
    </header>

    <main class="flex-1">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>

    <!-- AI气象助手 -->
    <AiAssistant />

    <footer class="border-t border-line bg-ink-alt px-6 py-10 text-center">
      <p class="text-sm text-dim">广西壮族自治区气象数据建模分析 · 2005—2025</p>
      <p class="mt-2 text-xs text-dim/60">
        数据来源：NASA POWER · NCEP/NCAR R1 · MODIS NDVI　|　工具：Python / PyTorch / scikit-learn / statsmodels
      </p>
    </footer>
  </div>
</template>
