/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'primary': {
          50: '#FFF5F5',
          100: '#FFCDD2',
          200: '#EF9A9A',
          300: '#E57373',
          400: '#EF5350',
          500: '#D32F2F',
          600: '#C62828',
          700: '#B71C1C',
          800: '#8E2424',
          900: '#6B1B1B',
        },
        'bio': {
          50: '#f0fdf4',
          100: '#dcfce7',
          200: '#bbf7d0',
          300: '#86efac',
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
          700: '#15803d',
          800: '#166534',
          900: '#14532d',
        },
        'surface': {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
          950: '#020617',
        },
        'accent': {
          50: '#E6F4FA',
          100: '#B3DEF2',
          200: '#80C8EA',
          300: '#4DB2E2',
          400: '#1A9CDA',
          500: '#0086C9',
          600: '#0077B6',
          700: '#006299',
          800: '#004D7A',
          900: '#00395C',
        },
      },
      fontFamily: {
        'display': ['"Source Serif 4"', 'Georgia', 'serif'],
        'sans': ['"IBM Plex Sans"', 'system-ui', 'sans-serif'],
        'mono': ['"JetBrains Mono"', 'Consolas', 'monospace'],
      },
      fontSize: {
        'display-lg': ['2rem', { lineHeight: '2.5rem', letterSpacing: '-0.02em', fontWeight: '700' }],
        'display-sm': ['1.25rem', { lineHeight: '1.75rem', letterSpacing: '-0.01em', fontWeight: '600' }],
      },
      boxShadow: {
        'subtle': '0 1px 2px 0 rgba(0, 0, 0, 0.03)',
        'card': '0 1px 3px 0 rgba(0, 0, 0, 0.06), 0 1px 2px -1px rgba(0, 0, 0, 0.04)',
        'elevated': '0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -2px rgba(0, 0, 0, 0.05)',
        'modal': '0 20px 60px -15px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(0, 0, 0, 0.05)',
        'glow-accent': '0 0 20px -5px rgba(0, 119, 182, 0.3)',
      },
      animation: {
        'fade-in-up': 'fade-in-up 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
        'scale-in': 'scale-in 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
        'shimmer': 'shimmer 2s ease-in-out infinite',
        'glow-pulse': 'glow-pulse 2s ease-in-out infinite',
      },
      keyframes: {
        'fade-in-up': {
          from: { opacity: '0', transform: 'translateY(16px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'scale-in': {
          from: { opacity: '0', transform: 'scale(0.95)' },
          to: { opacity: '1', transform: 'scale(1)' },
        },
        'shimmer': {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        'glow-pulse': {
          '0%, 100%': { boxShadow: '0 0 0 0 var(--agent-color, #0077B6)' },
          '50%': { boxShadow: '0 0 0 6px transparent' },
        },
      },
    },
  },
  plugins: [],
}
