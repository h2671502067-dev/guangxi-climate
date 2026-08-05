import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  // 相对路径 base：兼容 GitHub Pages 子路径（username.github.io/仓库名/）与根路径部署
  base: './',
  plugins: [vue(), tailwindcss()],
  server: {
    port: 8000,
    host: true,
  },
})
