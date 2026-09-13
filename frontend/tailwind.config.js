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
          bg: '#090c10',
          darker: '#06080c',
          panel: '#0d1117',
          surface: '#161b22',
          border: '#30363d',
          borderLight: '#484f58',
          accent: '#10b981', // radar green
          accentHover: '#059669',
          accentDim: 'rgba(16, 185, 129, 0.15)',
          warning: '#f59e0b',
          warningDim: 'rgba(245, 158, 11, 0.15)',
          cyan: '#06b6d4',
          cyanDim: 'rgba(6, 182, 212, 0.15)',
          red: '#ef4444',
          redDim: 'rgba(239, 68, 68, 0.15)',
          text: '#e6edf3',
          muted: '#8b949e',
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
