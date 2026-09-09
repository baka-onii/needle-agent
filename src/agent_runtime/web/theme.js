// Apply before styles render. No inline scripts or third-party dependencies.
(() => {
  let saved = null;
  try {
    saved = localStorage.getItem("needle-theme");
  } catch {}
  document.documentElement.dataset.theme = ["light", "dark"].includes(saved)
    ? saved
    : window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
})();
