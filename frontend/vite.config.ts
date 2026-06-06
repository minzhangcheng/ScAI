import { defineConfig, Plugin } from 'vite'
import react from '@vitejs/plugin-react'

const backendPort = /^\d+$/.test(process.env.SCAI_BACKPORT || '') ? process.env.SCAI_BACKPORT! : '8000'
const backendHost = /^[\w.\-]+$/.test(process.env.SCAI_HOST || '') ? process.env.SCAI_HOST! : 'localhost'

function disableRequestTimeout(): Plugin {
  return {
    name: 'disable-request-timeout',
    configureServer(server) {
      server.httpServer!.requestTimeout = 0
      server.httpServer!.headersTimeout = 0
    },
    configurePreviewServer(server) {
      server.httpServer!.requestTimeout = 0
      server.httpServer!.headersTimeout = 0
    },
  }
}

export default defineConfig({
  plugins: [react(), disableRequestTimeout()],
  server: {
    port: 3000,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: `http://${backendHost}:${backendPort}`,
        changeOrigin: true,
        proxyTimeout: 0,
        timeout: 86400000,
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
        changeOrigin: true,
        proxyTimeout: 0,
        timeout: 86400000,
      },
      '/ws': {
        target: `ws://${backendHost}:${backendPort}`,
        ws: true
      }
    }
  }
})
