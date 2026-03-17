import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backendPort = process.env.SCAI_BACKPORT || '8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 3000,
    allowedHosts: [
      'scai.cmz.ac.cn',
      'scai.organoids.org.cn'
    ],
    proxy: {
      '/api': {
        target: `http://localhost:${backendPort}`,
        changeOrigin: true
      },
      '/ws': {
        target: `ws://localhost:${backendPort}`,
        ws: true
      }
    }
  }
})
