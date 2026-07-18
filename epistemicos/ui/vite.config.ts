import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/claims': 'http://127.0.0.1:8000',
      '/events': 'http://127.0.0.1:8000',
      '/stream': 'http://127.0.0.1:8000',
      '/evidence': 'http://127.0.0.1:8000',
      '/retract': 'http://127.0.0.1:8000',
      '/explain': 'http://127.0.0.1:8000',
      '/metrics': 'http://127.0.0.1:8000',
      '/demo': 'http://127.0.0.1:8000',
    },
  },
})
