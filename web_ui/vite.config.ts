import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: process.env.HOST || '0.0.0.0',
    port: parseInt(process.env.PORT) || 50001,
    strictPort: true,  // 强制使用指定端口
    allowedHosts: [
      'localhost',
      '127.0.0.1',
      '0.0.0.0',
      'wufj1384754.bohrium.tech'
    ],
    cors: true,
    proxy: {
      '/api': {
        target: `http://${process.env.BACKEND_HOST || 'localhost'}:${process.env.BACKEND_PORT || 8000}`,
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path.replace(/^\/api/, '/api')
      },
      '/ws': {
        target: `ws://${process.env.BACKEND_HOST || 'localhost'}:${process.env.BACKEND_PORT || 8000}`,
        ws: true,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  resolve: {
    alias: {
      '@': '/src',
    },
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'antd'],
  },
})