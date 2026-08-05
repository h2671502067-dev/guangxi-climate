import { createApp } from 'vue'
import { createRouter, createWebHashHistory } from 'vue-router'
import './style.css'

import App from './App.vue'
import HomeView from './views/HomeView.vue'
import DashboardView from './views/DashboardView.vue'
import DroughtView from './views/DroughtView.vue'
import TrendView from './views/TrendView.vue'
import CorrelationView from './views/CorrelationView.vue'
import ExtremeView from './views/ExtremeView.vue'
import ClusteringView from './views/ClusteringView.vue'
import MethodsView from './views/MethodsView.vue'
import ConclusionView from './views/ConclusionView.vue'

const router = createRouter({
  history: createWebHashHistory(),
  scrollBehavior() {
    return { top: 0 }
  },
  routes: [
    { path: '/', component: HomeView, meta: { title: '首页' } },
    { path: '/dashboard', component: DashboardView, meta: { title: '数据分析' } },
    { path: '/drought', component: DroughtView, meta: { title: '干旱监测' } },
    { path: '/trend', component: TrendView, meta: { title: '气候趋势' } },
    { path: '/correlation', component: CorrelationView, meta: { title: '相关性' } },
    { path: '/extreme', component: ExtremeView, meta: { title: '极端事件' } },
    { path: '/clustering', component: ClusteringView, meta: { title: '城市聚类' } },
    { path: '/methods', component: MethodsView, meta: { title: '方法论' } },
    { path: '/conclusion', component: ConclusionView, meta: { title: '结论' } },
  ],
})

router.afterEach((to) => {
  document.title = `${to.meta.title || '首页'} · 广西气象数据建模分析`
})

createApp(App).use(router).mount('#app')
