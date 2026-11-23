// Hand-drawn SVG charts (no chart library: the app works offline).
const NS = "http://www.w3.org/2000/svg";

function s(tag, attrs = {}, text) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, String(v));
  if (text != null) el.textContent = String(text);
  return el;
}

function band(pct) {
  return pct >= 80 ? "good" : pct >= 50 ? "warn" : "bad";
}

// Accuracy bars (0-100 %) per row, in HTML so the labels stay readable at any width: rows = [{label, value, detail}]
export function accuracyBars(rows, { title } = {}) {
  const wrap = document.createElement("div");
  wrap.className = "bars";
  wrap.setAttribute("role", "list");
  wrap.setAttribute("aria-label", title || "Accuracy");
  for (const r of rows) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.setAttribute("role", "listitem");
    row.title = `${r.label}: ${Math.round(r.value)}%${r.detail ? ` (${r.detail} correct)` : ""}`;
    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = r.label;
    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = `bar-fill ${band(r.value)}`;
    fill.style.width = "0%";
    track.append(fill);
    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = `${Math.round(r.value)}%`;
    const detail = document.createElement("span");
    detail.className = "bar-detail";
    detail.textContent = r.detail || "";
    row.append(label, track, value, detail);
    wrap.append(row);
    requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${Math.max(1.5, r.value)}%`; }));
  }
  return wrap;
}

// Score over time: points = [{label, value}] (value in %)
export function trendLine(points, { title } = {}) {
  const w = 640, hgt = 220, padL = 36, padR = 14, padT = 14, padB = 30;
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${w} ${hgt}`, role: "img", "aria-label": title || "Score trend" });
  const iw = w - padL - padR, ih = hgt - padT - padB;
  for (const t of [0, 25, 50, 75, 100]) {
    const y = padT + ih - (ih * t) / 100;
    svg.append(s("line", { class: "grid-line", x1: padL, x2: w - padR, y1: y, y2: y }));
    svg.append(s("text", { x: padL - 8, y: y + 4, "text-anchor": "end" }, t));
  }
  if (!points.length) return svg;
  const x = (i) => padL + (points.length === 1 ? iw / 2 : (iw * i) / (points.length - 1));
  const y = (v) => padT + ih - (ih * v) / 100;
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(p.value).toFixed(1)}`).join(" ");
  if (points.length > 1) {
    svg.append(s("path", { class: "area", d: `${d} L${x(points.length - 1)} ${padT + ih} L${x(0)} ${padT + ih} Z` }));
    svg.append(s("path", { class: "line", d }));
  }
  const step = Math.ceil(points.length / 8);
  points.forEach((p, i) => {
    const g = s("g");
    g.append(s("title", {}, `${p.label}: ${p.value}%`));
    g.append(s("circle", { class: "point", cx: x(i), cy: y(p.value), r: 4.5 }));
    svg.append(g);
    if (i % step === 0 || i === points.length - 1) svg.append(s("text", { x: x(i), y: hgt - 8, "text-anchor": "middle" }, i + 1));
  });
  return svg;
}

// Circular score gauge
export function scoreRing(percent) {
  const r = 64, c = 2 * Math.PI * r;
  const wrap = document.createElement("div");
  wrap.className = "score-ring";
  const svg = s("svg", { viewBox: "0 0 150 150", "aria-hidden": "true" });
  svg.append(s("circle", { class: "ring-bg", cx: 75, cy: 75, r, fill: "none", "stroke-width": 12 }));
  const fg = s("circle", {
    class: "ring-fg", cx: 75, cy: 75, r, fill: "none", "stroke-width": 12, "stroke-linecap": "round",
    "stroke-dasharray": c.toFixed(1), "stroke-dashoffset": c.toFixed(1),
  });
  fg.style.stroke = `var(--${band(percent)})`;
  svg.append(fg);
  wrap.append(svg);
  const label = document.createElement("div");
  label.className = "ring-label";
  const inner = document.createElement("div");
  const b = document.createElement("b");
  b.textContent = `${Math.round(percent)}%`;
  const sp = document.createElement("span");
  sp.textContent = "score";
  inner.append(b, sp);
  label.append(inner);
  wrap.append(label);
  requestAnimationFrame(() => requestAnimationFrame(() => fg.setAttribute("stroke-dashoffset", (c * (1 - percent / 100)).toFixed(1))));
  return wrap;
}
