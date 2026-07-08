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
        // UNO card colors — vibrant and game-authentic
        uno: {
          red: '#E53935',
          yellow: '#FDD835',
          green: '#43A047',
          blue: '#1E88E5',
          black: '#1A1A2E',
        },
        // UI palette — premium dark mode
        surface: {
          50: '#f8f9fc',
          100: '#eef1f6',
          200: '#dde3ed',
          300: '#c4cdd9',
          400: '#8d99ae',
          500: '#6b7a8d',
          600: '#4a5568',
          700: '#2d3748',
          800: '#1a202c',
          900: '#0f1219',
          950: '#080a0f',
        },
        accent: {
          DEFAULT: '#6C63FF',
          light: '#8B83FF',
          dark: '#4D45E6',
          glow: 'rgba(108, 99, 255, 0.3)',
        },
        success: '#10B981',
        warning: '#F59E0B',
        danger: '#EF4444',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['Outfit', 'Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'card-deal': 'cardDeal 0.5s ease-out',
        'card-play': 'cardPlay 0.4s ease-in-out',
        'card-draw': 'cardDraw 0.3s ease-out',
        'pulse-glow': 'pulseGlow 2s infinite',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'fade-in': 'fadeIn 0.3s ease-out',
        'spin-slow': 'spin 3s linear infinite',
        'bounce-subtle': 'bounceSubtle 0.6s ease-in-out',
        'shimmer': 'shimmer 2s linear infinite',
      },
      keyframes: {
        cardDeal: {
          '0%': { transform: 'translateY(-100px) rotate(-20deg) scale(0.5)', opacity: '0' },
          '100%': { transform: 'translateY(0) rotate(0) scale(1)', opacity: '1' },
        },
        cardPlay: {
          '0%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.1) translateY(-10px)' },
          '100%': { transform: 'scale(0.9) translateY(0)' },
        },
        cardDraw: {
          '0%': { transform: 'translateX(50px) scale(0.8)', opacity: '0' },
          '100%': { transform: 'translateX(0) scale(1)', opacity: '1' },
        },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 5px rgba(108, 99, 255, 0.3)' },
          '50%': { boxShadow: '0 0 20px rgba(108, 99, 255, 0.6)' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        bounceSubtle: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-5px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
      boxShadow: {
        'card': '0 4px 20px rgba(0, 0, 0, 0.3)',
        'card-hover': '0 8px 30px rgba(0, 0, 0, 0.4)',
        'glow-accent': '0 0 20px rgba(108, 99, 255, 0.4)',
        'glow-red': '0 0 20px rgba(229, 57, 53, 0.4)',
        'glow-blue': '0 0 20px rgba(30, 136, 229, 0.4)',
        'glow-green': '0 0 20px rgba(67, 160, 71, 0.4)',
        'glow-yellow': '0 0 20px rgba(253, 216, 53, 0.4)',
      },
    },
  },
  plugins: [],
}
