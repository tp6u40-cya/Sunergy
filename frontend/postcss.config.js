

// postcss.config.js (這是對的)
module.exports = {
  plugins: {
    '@tailwindcss/postcss': {}, // <--- 這才是正確的
    autoprefixer: {},
  },
};