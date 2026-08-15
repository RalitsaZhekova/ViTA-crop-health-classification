"use strict";

(() => {
  let savedTheme = null;
  try {
    savedTheme = window.localStorage.getItem("vita-theme");
  } catch (_error) {
    // Storage can be disabled; the system preference remains a safe default.
  }
  const theme = ["light", "dark"].includes(savedTheme)
    ? savedTheme
    : (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
})();
