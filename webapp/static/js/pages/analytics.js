import { get } from "../api.js";
import { accuracyBars, trendLine } from "../charts.js";
import { go } from "../nav.js";
import { alertBox, btn, clear, emptyState, fmtDate, fmtDuration, h, pageHead } from "../ui.js";

export async function render(root) {
  root.append(pageHead("Analytics", "How you are doing, based only on your real answers."));
  const filter = h("select", { "aria-label": "Quiz", style: { maxWidth: "340px" } }, h("option", { value: "" }, "All quizzes"));
  const body = h("div", { class: "stack" });
  root.append(h("div", { class: "row", style: { marginBottom: "1rem" } }, filter), body);

  const load = async () => {
    clear(body).append(h("div", { class: "skeleton", style: { height: "120px" } }));
    let d;
    try { d = await get(`/analytics${filter.value ? `?quiz_id=${encodeURIComponent(filter.value)}` : ""}`); }
    catch (err) { clear(body).append(alertBox("bad", err.message)); return; }
    if (filter.options.length === 1) for (const q of d.quizzes) filter.append(h("option", { value: q.id }, q.title));
    clear(body);
    if (!d.summary.attempts) {
      body.append(emptyState("chart", "No results yet", "Take a quiz and your scores, strengths and weak topics show up here.",
        btn("Take a quiz", { kind: "primary", icon: "play", onClick: () => go("quiz") })));
      return;
    }
    const s = d.summary;
    const pct = (v) => (v == null ? "-" : `${Math.round(v)}%`);
    body.append(h("div", { class: "grid cols-4" },
      tile("Attempts", s.attempts), tile("Average score", pct(s.average)), tile("Best score", pct(s.best)),
      tile("Answer accuracy", pct(s.accuracy), `${s.questions_answered.toLocaleString()} questions answered`)));
    body.append(h("section", { class: "card" }, h("div", { class: "card-head" }, h("h2", null, "Score over time"), h("span", { class: "sub" }, "attempt number →")),
      trendLine(d.trend.map((t) => ({ label: `${t.quiz} · ${fmtDate(t.at)}`, value: t.percent })), { title: "Score per attempt" })));
    body.append(h("div", { class: "grid cols-2" },
      h("section", { class: "card" }, h("h2", { style: { marginBottom: ".8rem" } }, "Accuracy by topic"),
        accuracyBars(d.by_topic.sort((a, b) => a.accuracy - b.accuracy).slice(0, 12).map((r) => ({ label: r.topic, value: r.accuracy, detail: `${r.correct}/${r.total}` })), { title: "Accuracy by topic" })),
      h("section", { class: "card" }, h("h2", { style: { marginBottom: ".8rem" } }, "Accuracy by difficulty"),
        accuracyBars(d.by_difficulty.map((r) => ({ label: r.difficulty, value: r.accuracy, detail: `${r.correct}/${r.total}` })), { title: "Accuracy by difficulty" }),
        h("h3", { style: { margin: "1.2rem 0 .5rem" } }, "Recommendations"),
        h("ul", { style: { margin: 0, paddingLeft: "1.1rem" } }, d.recommendations.map((r) => h("li", { class: "small", style: { marginBottom: ".3rem" } }, r))))));
    body.append(h("section", { class: "card" }, h("h2", { style: { marginBottom: ".6rem" } }, "Recent attempts"),
      h("div", { class: "table-wrap" }, h("table", { class: "table" },
        h("thead", null, h("tr", null, h("th", null, "Quiz"), h("th", null, "Date"), h("th", { class: "num" }, "Score"), h("th", { class: "num" }, "Correct"), h("th", { class: "num" }, "Time"))),
        h("tbody", null, d.attempts.map((a) => h("tr", null, h("td", null, a.quiz), h("td", null, fmtDate(a.at)),
          h("td", { class: "num" }, h("b", null, `${Math.round(a.percent)}%`)), h("td", { class: "num" }, `${a.correct}/${a.total}`),
          h("td", { class: "num" }, fmtDuration(a.duration_s)))))))));
  };
  filter.addEventListener("change", load);
  await load();
}

function tile(label, value, hint) {
  return h("div", { class: "card stat" }, h("span", { class: "label" }, label), h("span", { class: "value" }, String(value)), hint ? h("span", { class: "hint" }, hint) : null);
}
