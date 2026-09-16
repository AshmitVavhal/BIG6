/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        sat: {
          bg: '#090a0d',
          darker: '#060709',
          panel: '#101217',
          surface: '#161920',
          surfaceHover: '#1c202a',
          border: '#1f242e',
          borderLight: '#2b3240',
          accent: '#10b981', // radar green
          accentHover: '#059669',
          accentDim: 'rgba(16, 185, 129, 0.12)',
          warning: '#f59e0b',
          warningDim: 'rgba(245, 158, 11, 0.12)',
          cyan: '#06b6d4',
          cyanDim: 'rgba(6, 182, 212, 0.12)',
          red: '#ef4444',
          redDim: 'rgba(239, 68, 68, 0.12)',
          text: '#e6edf3',
          textSecondary: '#94a3b8',
          muted: '#64748b',
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Menlo', 'Consolas', 'Courier New', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      }
    },
  },
  plugins: [],
}

