import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backendPort = /^\d+$/.test(process.env.SCAI_BACKPORT || '') ? process.env.SCAI_BACKPORT! : '8000'
const backendHost = /^[\w.\-]+$/.test(process.env.SCAI_HOST || '') ? process.env.SCAI_HOST! : 'localhost'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    allowedHosts: true,
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
  },
  preview: {
    port: 3000,
    allowedHosts: true,
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
