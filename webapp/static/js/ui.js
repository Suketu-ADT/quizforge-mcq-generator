// Small DOM toolkit. Everything that came from the model or the user is inserted as text (never as HTML).

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k === "style") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "value") el.value = v;
    else if (k === "checked") el.checked = !!v;
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, String(v));
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false || c === "") continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function clear(el) {
  while (el.firstChild) el.firstChild.remove();
  return el;
}

// ------------------------------------------------------------------------------------------------ icons
// Static, trusted SVG paths (24x24, stroke icons).
const PATHS = {
  home: "M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z",
  sparkles: "M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8zM19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9zM5 15l.7 1.6L7.3 17.3 5.7 18 5 19.6 4.3 18 2.7 17.3l1.6-.7z",
  library: "M4 5h4v15H4zM10 5h4v15h-4zM16.5 5.5l3.8-1 3.2 14.5-3.8 1z",
  play: "M8 5.5v13l11-6.5z",
  chart: "M4 20V10M10 20V4M16 20v-7M22 20H2",
  settings: "M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7zM19.4 13.5a7.7 7.7 0 0 0 0-3l2-1.6-2-3.4-2.4 1a7.6 7.6 0 0 0-2.6-1.5L14 2.5h-4l-.4 2.5a7.6 7.6 0 0 0-2.6 1.5l-2.4-1-2 3.4 2 1.6a7.7 7.7 0 0 0 0 3l-2 1.6 2 3.4 2.4-1a7.6 7.6 0 0 0 2.6 1.5l.4 2.5h4l.4-2.5a7.6 7.6 0 0 0 2.6-1.5l2.4 1 2-3.4z",
  info: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 16v-5M12 8h.01",
  upload: "M12 16V4M7 9l5-5 5 5M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3",
  file: "M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8zM14 3v5h5",
  x: "M6 6l12 12M18 6 6 18",
  check: "M5 12.5l4.5 4.5L19 7",
  alert: "M12 3 2 20h20zM12 10v4M12 17h.01",
  edit: "M4 20h4L19 9l-4-4L4 16zM13.5 6.5l4 4",
  refresh: "M20 11a8 8 0 0 0-14.7-4.3L4 8M4 4v4h4M4 13a8 8 0 0 0 14.7 4.3L20 16M20 20v-4h-4",
  trash: "M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v6M14 11v6",
  plus: "M12 5v14M5 12h14",
  download: "M12 4v12M7 11l5 5 5-5M4 20h16",
  copy: "M8 8h11v13H8zM5 16H4V3h11v1",
  search: "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zM21 21l-5-5",
  flag: "M5 21V4M5 4h11l-2 4 2 4H5",
  stop: "M7 7h10v10H7z",
  eye: "M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12zM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
  eyeOff: "M3 3l18 18M10.6 5.1A10 10 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-3 3.9M6.6 6.6A16.5 16.5 0 0 0 2 12s3.5 7 10 7a9.8 9.8 0 0 0 5.4-1.6M9.9 9.9a3 3 0 0 0 4.2 4.2",
  sun: "M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zM12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4",
  moon: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
  menu: "M4 6h16M4 12h16M4 18h16",
  chevronLeft: "M15 6l-6 6 6 6",
  chevronRight: "M9 6l6 6-6 6",
  clock: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 7v5l3 2",
  shield: "M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z",
  cpu: "M7 7h10v10H7zM9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3",
  dots: "M5 12h.01M12 12h.01M19 12h.01",
  target: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zM12 13a1 1 0 1 0 0-2 1 1 0 0 0 0 2z",
  bolt: "M13 2 4 14h7l-1 8 9-12h-7z",
  rotate: "M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5",
};

export function icon(name, cls = "") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", name === "play" || name === "stop" ? "currentColor" : "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.9");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", `ic ${cls}`.trim());
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", PATHS[name] || PATHS.info);
  svg.append(path);
  return svg;
}

// --------------------------------------------------------------------------------------------- widgets
export function btn(label, opts = {}) {
  const { kind = "", icon: ic, iconRight, onClick, title, disabled, type = "button", size = "" } = opts;
  return h("button", { class: `btn ${kind} ${size}`.trim(), type, title, disabled: !!disabled, onclick: onClick, "aria-label": !label && title ? title : null },
    ic ? icon(ic) : null, label, iconRight ? icon(iconRight) : null);
}

export async function busy(button, work) {
  if (button.classList.contains("loading")) return;
  button.classList.add("loading");
  button.disabled = true;
  try { return await work(); }
  finally { button.classList.remove("loading"); button.disabled = false; }
}

export function alertBox(kind, ...content) {
  const ic = { info: "info", warn: "alert", bad: "alert", good: "check" }[kind] || "info";
  return h("div", { class: `alert ${kind}`, role: kind === "bad" ? "alert" : null }, icon(ic), h("div", { class: "alert-body" }, ...content));
}

export function tag(text, kind = "") {
  return h("span", { class: `tag ${kind}`.trim() }, text);
}

export function difficultyTag(d) {
  return tag(d, (d || "").toLowerCase());
}

export function seg(options, value, onChange, { label } = {}) {
  const wrap = h("div", { class: "seg", role: "group", "aria-label": label });
  const render = (v) => {
    for (const b of wrap.children) b.setAttribute("aria-pressed", String(b.dataset.value === String(v)));
  };
  for (const o of options) {
    const opt = typeof o === "object" ? o : { value: o, label: String(o) };
    wrap.append(h("button", {
      type: "button", dataset: { value: String(opt.value) }, disabled: !!opt.disabled, title: opt.title,
      onclick: () => { render(opt.value); onChange(opt.value); },
    }, opt.label));
  }
  render(value);
  wrap.setValue = render;
  return wrap;
}

export function toggle(label, checked, onChange, help) {
  const input = h("input", { type: "checkbox", checked, onchange: (e) => onChange(e.target.checked) });
  return h("label", { class: "switch" }, input, h("span", null, label, help ? h("span", { class: "help" }, help) : null));
}

export function field(label, control, help) {
  return h("label", { class: "field" }, h("span", null, label), control, help ? h("span", { class: "help" }, help) : null);
}

export function stepper(value, min, max, onChange) {
  const out = h("output", null, String(value));
  const set = (v) => { value = Math.max(min, Math.min(max, v)); out.textContent = String(value); onChange(value); };
  return h("div", { class: "stepper" },
    h("button", { type: "button", "aria-label": "Fewer", onclick: () => set(value - 1) }, "−"), out,
    h("button", { type: "button", "aria-label": "More", onclick: () => set(value + 1) }, "+"));
}

// A small dropdown menu: items = [{label, icon, onClick} | "-" | {heading}]
export function menu(trigger, items, { up = false } = {}) {
  const wrap = h("div", { class: "menu" }, trigger);
  let pop = null;
  const close = () => { pop?.remove(); pop = null; document.removeEventListener("click", outside, true); };
  const outside = (e) => { if (!wrap.contains(e.target)) close(); };
  trigger.setAttribute("aria-haspopup", "menu");
  trigger.addEventListener("click", (e) => {
    e.stopPropagation();
    if (pop) return close();
    pop = h("div", { class: `menu-pop ${up ? "up" : ""}`, role: "menu" });
    for (const it of items) {
      if (it === "-") pop.append(h("hr"));
      else if (it.heading) pop.append(h("div", { class: "menu-label" }, it.heading));
      else pop.append(h("button", { type: "button", role: "menuitem", onclick: () => { close(); it.onClick(); } },
        it.icon ? icon(it.icon) : null, it.label));
    }
    wrap.append(pop);
    document.addEventListener("click", outside, true);
    pop.querySelector("button")?.focus();
  });
  wrap.addEventListener("keydown", (e) => { if (e.key === "Escape") { close(); trigger.focus(); } });
  return wrap;
}

// ------------------------------------------------------------------------------------- toasts & modals
export function toast(message, kind = "", ms = 4200) {
  const box = document.getElementById("toasts");
  const t = h("div", { class: `toast ${kind}` }, icon(kind === "bad" ? "alert" : kind === "warn" ? "alert" : "check"), h("div", null, message));
  box.append(t);
  setTimeout(() => { t.classList.add("leaving"); setTimeout(() => t.remove(), 260); }, ms);
}

export function confirmDialog({ title, message, confirm = "Confirm", danger = false, cancel = "Cancel" }) {
  return new Promise((resolve) => {
    const dlg = h("dialog", { class: "modal", "aria-labelledby": "dlg-title" });
    const done = (v) => { dlg.close(); dlg.remove(); resolve(v); };
    dlg.append(
      h("div", { class: "modal-body" }, h("h2", { id: "dlg-title" }, title),
        ...(Array.isArray(message) ? message : [h("p", null, message)])),
      h("div", { class: "modal-foot" },
        btn(cancel, { kind: "ghost", onClick: () => done(false) }),
        btn(confirm, { kind: danger ? "danger solid" : "primary", onClick: () => done(true) })),
    );
    dlg.addEventListener("cancel", (e) => { e.preventDefault(); done(false); });
    document.body.append(dlg);
    dlg.showModal();
    dlg.querySelector(".modal-foot .btn:last-child").focus();
  });
}

export function promptDialog({ title, label, value = "", confirm = "Save" }) {
  return new Promise((resolve) => {
    const input = h("input", { type: "text", value });
    const dlg = h("dialog", { class: "modal" });
    const done = (v) => { dlg.close(); dlg.remove(); resolve(v); };
    const form = h("form", { method: "dialog", onsubmit: (e) => { e.preventDefault(); done(input.value.trim() || null); } },
      h("div", { class: "modal-body" }, h("h2", null, title), field(label, input)),
      h("div", { class: "modal-foot" }, btn("Cancel", { kind: "ghost", onClick: () => done(null) }), btn(confirm, { kind: "primary", type: "submit" })));
    dlg.append(form);
    dlg.addEventListener("cancel", (e) => { e.preventDefault(); done(null); });
    document.body.append(dlg);
    dlg.showModal();
    input.select();
  });
}

// ------------------------------------------------------------------------------------------ formatters
export function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

export function fmtWhen(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const s = (Date.now() - d.getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 7 * 86400) return `${Math.floor(s / 86400)} d ago`;
  return fmtDate(iso);
}

export function fmtDuration(sec) {
  if (sec == null) return "-";
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60);
  return m >= 60 ? `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m` : `${m}:${String(s % 60).padStart(2, "0")}`;
}

export function plural(n, word, many) {
  return `${n.toLocaleString()} ${n === 1 ? word : (many || word + "s")}`;
}

export function debounce(fn, ms) {
  let t;
  const d = (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  d.flush = (...args) => { clearTimeout(t); fn(...args); };
  return d;
}

export function pageHead(title, subtitle, ...actions) {
  return h("div", { class: "page-head" },
    h("div", null, h("h1", null, title), subtitle ? h("p", null, subtitle) : null),
    actions.length ? h("div", { class: "row" }, ...actions) : null);
}

export function emptyState(ic, title, text, ...actions) {
  return h("div", { class: "card empty" }, icon(ic), h("h3", null, title), h("p", { class: "muted" }, text),
    actions.length ? h("div", { class: "row", style: { justifyContent: "center" } }, ...actions) : null);
}
