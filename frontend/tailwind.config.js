// tailwind.config.js
module.exports = {
  darkMode: "class",
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html" // <--- 關鍵！把這行加回來
  ],
  theme: {
    extend: {
      colors: {
        "primary": "#f2cc0d",
        "background-light": "#f8f8f5",
        "background-dark": "#121212",
      },
      fontFamily: {
        // ... (您原有的字體設定)
        display: ['Inter', '"Noto Sans TC"', 'sans-serif'],
      },
      borderRadius: {
        // ... (您原有的圓角設定)
      },
      keyframes: {
        // ... (您剛才加入的動畫) ...
        sunRise: {
          '0%': { transform: 'translateY(100%) scale(0.8)', opacity: '0.8' },
          '50%': { transform: 'translateY(0%) scale(1)', opacity: '1' },
          '100%': { transform: 'translateY(0%) scale(1)', opacity: '1' },
        },
        fadeInUp: {
          'from': { opacity: '0', transform: 'translateY(20px)' },
          'to': { opacity: '1', transform: 'translateY(0)' },
        },
        bounceOnce: {
          '0%, 100%': { transform: 'translateY(0)' },
          '20%': { transform: 'translateY(-8px)' },
          '40%': { transform: 'translateY(0)' },
          '60%': { transform: 'translateY(-4px)' },
          '80%': { transform: 'translateY(0)' },
        },
        bounceHorizontal: {
          '0%, 100%': { transform: 'translateX(0)' },
          '20%': { transform: 'translateX(4px)' },
          '40%': { transform: 'translateX(0)' },
        },
      },
      animation: {
        
        'sun-rise': 'sunRise 2s ease-out forwards',
        'fade-in-up': 'fadeInUp 1s ease-out forwards',
        'bounce-once': 'bounceOnce 1s ease-out forwards',
        'bounce-horizontal': 'bounceHorizontal 1.5s ease-in-out infinite',
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/container-queries'),
  ],
};