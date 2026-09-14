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
          bg: '#080808',
          darker: '#000000',
          panel: '#101010',
          surface: '#171717',
          border: '#242424',
          borderLight: '#333333',
          accent: '#10b981', // radar green
          accentHover: '#059669',
          accentDim: 'rgba(16, 185, 129, 0.12)',
          warning: '#f59e0b',
          warningDim: 'rgba(245, 158, 11, 0.12)',
          cyan: '#06b6d4',
          cyanDim: 'rgba(6, 182, 212, 0.12)',
          red: '#ef4444',
          redDim: 'rgba(239, 68, 68, 0.12)',
          text: '#ededed',
          muted: '#737373',
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
