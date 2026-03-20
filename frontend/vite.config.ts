import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backendPort = /^\d+$/.test(process.env.SCAI_BACKPORT || '') ? process.env.SCAI_BACKPORT! : '8000'
const backendHost = /^[\w.\-]+$/.test(process.env.SCAI_HOST || '') ? process.env.SCAI_HOST! : 'localhost'
const extraHosts = (process.env.SCAI_ALLOWED_HOSTS || '').split(',').filter(Boolean)

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    allowedHosts: [
      'scai.cmz.ac.cn',
      'scai.organoids.org.cn',
      ...extraHosts
    ],
    proxy: {
      '/api': {
        target: `http://${backendHost}:${backendPort}`,
        changeOrigin: true
      },
      '/ws': {
        target: `ws://${backendHost}:${backendPort}`,
        ws: true
      }
    }
  }
})
