/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./templates/**/*.html",
    "./static/src/**/*.js",
    "./node_modules/flowbite/**/*.js"
  ],
  theme: {
    extend: {},
  },
  daisyui: {
    themes: [
      {
        mytheme: {
          "primary": "#f5f5f4",
          "secondary": "#be123c",
          "accent": "#a8a29e",
          "neutral": "#44403c",
          "base-100": "#292524",
          "info": "#155e75",
          "success": "#0d9488",
          "warning": "#d97706",
          "error": "#780000",
        },
      },
    ],
  },
  plugins: [
    require("daisyui"), 
  ],
}

