// QuizForge front end: hash router, sidebar, theme and the live model status.
import { clear, h, icon } from "./ui.js";
import { go } from "./nav.js";
import { onStatus, refreshStatus } from "./store.js";
import * as dashboard from "./pages/dashboard.js";
import * as generate from "./pages/generate.js";
import * as bank from "./pages/bank.js";
import * as quiz from "./pages/quiz.js";
import * as analytics from "./pages/analytics.js";
import * as settings from "./pages/settings.js";
import * as about from "./pages/about.js";

const ROUTES = [
  { path: "dashboard", label: "Dashboard", icon: "home", page: dashboard },
  { path: "generate", label: "Generate Quiz", icon: "sparkles", page: generate },
  { path: "bank", label: "Question Bank", icon: "library", page: bank },
  { path: "quiz", label: "Take a Quiz", icon: "play", page: quiz },
  { path: "analytics", label: "Analytics", icon: "chart", page: analytics },
  { path: "settings", label: "Settings", icon: "settings", page: settings },
  { path: "about", label: "About", icon: "info", page: about },
];

const main = document.getElementById("main");
const nav = document.getElementById("nav");
const sidebar = document.getElementById("sidebar");
const scrim = document.getElementById("scrim");
const menuBtn = document.getElementById("menu-btn");
let cleanup = null;

// ---------------------------------------------------------------------------------------------- theme
const THEME_KEY = "quizforge-theme";
function storedTheme() {
  try { return localStorage.getItem(THEME_KEY); } catch { return null; }
}
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = document.getElementById("theme-btn");
  clear(btn).append(icon(theme === "dark" ? "sun" : "moon"));
  btn.title = theme === "dark" ? "Switch to the light theme" : "Switch to the dark theme";
}
applyTheme(storedTheme() || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
document.getElementById("theme-btn").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  try { localStorage.setItem(THEME_KEY, next); } catch { /* private mode: the theme just is not remembered */ }
  applyTheme(next);
});

// ----------------------------------------------------------------------------------------- navigation
function buildNav() {
  for (const r of ROUTES) {
    nav.append(h("a", { href: `#/${r.path}`, dataset: { path: r.path } }, icon(r.icon), h("span", null, r.label)));
  }
}

function setMenu(open) {
  sidebar.classList.toggle("open", open);
  scrim.hidden = !open;
  menuBtn.setAttribute("aria-expanded", String(open));
}
menuBtn.append(icon("menu"));
menuBtn.addEventListener("click", () => setMenu(!sidebar.classList.contains("open")));
scrim.addEventListener("click", () => setMenu(false));

function parseHash() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent);
  return { name: parts[0] || "dashboard", params: parts.slice(1) };
}

async function route() {
  const { name, params } = parseHash();
  const r = ROUTES.find((x) => x.path === name) || ROUTES[0];
  if (typeof cleanup === "function") {
    try { cleanup(); } catch { /* ignore */ }
  }
  cleanup = null;
  for (const a of nav.querySelectorAll("a")) a.classList.toggle("active", a.dataset.path === r.path);
  document.getElementById("topbar-title").textContent = r.label;
  document.title = `${r.label} · QuizForge`;
  setMenu(false);
  clear(main);
  window.scrollTo(0, 0);
  try {
    cleanup = await r.page.render(main, params);
  } catch (err) {
    console.error(err);
    clear(main).append(h("div", { class: "card" }, h("h2", null, "Something went wrong"), h("p", { class: "muted" }, String(err.message || err))));
  }
  main.focus({ preventScroll: true });
}

// ------------------------------------------------------------------------------------------- status
function paintStatus(st) {
  const pill = document.getElementById("status-pill");
  const label = pill.querySelector(".label");
  pill.classList.remove("ok", "warn", "bad");
  if (st.server_down) { pill.classList.add("bad"); label.textContent = "App server stopped"; }
  else if (!st.connected) { pill.classList.add("bad"); label.textContent = "Ollama not running"; }
  else if (!st.ready) { pill.classList.add("warn"); label.textContent = "Model not installed"; }
  else { pill.classList.add("ok"); label.textContent = `Ready · ${st.model_name}`; }
  const chip = document.getElementById("model-chip");
  chip.hidden = !st.connected;
  chip.textContent = st.model_name || "";
  chip.classList.toggle("tuned", (st.model_name || "").startsWith("Tuned"));
  chip.title = "The model that writes your questions (change it in Settings)";
}
document.getElementById("status-pill").addEventListener("click", () => go("settings"));

window.addEventListener("scroll", () => document.querySelector(".topbar").classList.toggle("scrolled", window.scrollY > 4), { passive: true });
window.addEventListener("hashchange", route);
window.addEventListener("focus", () => refreshStatus());

buildNav();
onStatus(paintStatus);
refreshStatus();
setInterval(() => { if (!document.hidden) refreshStatus(); }, 20000);
route();
