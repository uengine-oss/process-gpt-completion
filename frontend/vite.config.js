// vue-cli(지원 종료, 개발 도구 취약점 다수)에서 vite 로 이전 — 화면 코드는 그대로
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()]
})
