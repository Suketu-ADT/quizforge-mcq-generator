import { del, download, get, post, put } from "../api.js";
import { accuracyBars, scoreRing } from "../charts.js";
import { go, takeHandOff } from "../nav.js";
import {
  alertBox, btn, clear, confirmDialog, debounce, difficultyTag, emptyState, field, fmtDuration, h, icon, menu, pageHead, plural, seg,
  tag, toast,
} from "../ui.js";

const PREFS_KEY = "quizforge-quiz-prefs";
let R = null; // the running quiz, kept while you visit other pages
let runCleanup = null; // stops the timer and key handler of the quiz on screen

function prefs() {
  try { return { mode: "one", instant: true, timer: "off", minutes: 10, ...JSON.parse(localStorage.getItem(PREFS_KEY) || "{}") }; }
  catch { return { mode: "one", instant: true, timer: "off", minutes: 10 }; }
}
function savePrefs(p) {
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch { /* not remembered */ }
}

const persist = debounce(() => {
  if (!R || R.subset || R.result) return;
  put("/session", { session: toState() }).catch(() => {});
}, 400);

function toState() {
  return { quiz_id: R.quiz.id, answers: R.answers, flagged: [...R.flagged], current: R.current, elapsed_s: Math.round(elapsed() * 10) / 10,
    time_limit_s: R.timer === "countdown" ? R.limit : null, instant_feedback: R.instant, timed: R.timer !== "off", mode: R.mode };
}

function elapsed() {
  return R.elapsedBefore + (R.result ? 0 : (Date.now() - R.startedAt) / 1000);
}

function begin(quiz, opts, saved) {
  R = {
    quiz, mode: opts.mode, instant: opts.instant, timer: opts.timer, limit: opts.timer === "countdown" ? Math.max(1, opts.minutes) * 60 : null,
    answers: saved?.answers || {}, flagged: new Set(saved?.flagged || []), current: saved?.current || 0,
    startedAt: Date.now(), elapsedBefore: saved?.elapsed_s || 0, subset: opts.subset || null, result: null, review: "all",
  };
  if (opts.subset) R.quiz = { ...quiz, questions: quiz.questions.filter((q) => opts.subset.includes(q.id)) };
  persist.flush();
}

export async function render(root, params) {
  const wanted = params[0];
  if (wanted === "resume") {
    try {
      const s = (await get("/session")).session;
      if (s) {
        const quiz = (await get(`/quizzes/${s.quiz_id}`)).quiz;
        const mode = s.mode === "all" || /all/i.test(s.mode || "") ? "all" : "one";
        begin(quiz, { mode, instant: !!s.instant_feedback, timer: s.time_limit_s ? "countdown" : s.timed ? "stopwatch" : "off", minutes: (s.time_limit_s || 600) / 60 }, s);
        if (R.limit) R.limit = s.time_limit_s;
      }
    } catch (err) { toast(err.message, "bad"); }
    history.replaceState(null, "", "#/quiz");
  }
  const quick = takeHandOff("start-quiz");
  if (quick) {
    try { begin((await get(`/quizzes/${quick}`)).quiz, prefs()); } catch (err) { toast(err.message, "bad"); }
  }
  const stop = () => { runCleanup?.(); runCleanup = null; };
  if (R && (!wanted || wanted === "resume" || wanted === R.quiz.id || quick)) {
    if (R.result) renderResults(root); else renderRun(root);
  } else await renderSetup(root, wanted === "resume" ? null : wanted);
  return stop;
}

// ------------------------------------------------------------------------------------------------ setup
async function renderSetup(root, preselect) {
  root.append(pageHead("Take a Quiz", "Pick a quiz from your bank and choose how you want to take it."));
  let list, session;
  try { [list, session] = await Promise.all([get("/quizzes"), get("/session").then((r) => r.session).catch(() => null)]); }
  catch (err) { root.append(alertBox("bad", err.message)); return; }
  if (!list.quizzes.length) {
    root.append(emptyState("library", "No saved quizzes yet", "Generate a quiz and save it to your question bank first.",
      btn("Generate a quiz", { kind: "primary", icon: "sparkles", onClick: () => go("generate") })));
    return;
  }
  if (session) {
    root.append(h("div", { style: { marginBottom: "1rem" } }, alertBox("info",
      h("b", null, `Unfinished: “${session.quiz_title}”`), ` — ${Object.keys(session.answers || {}).length} of ${session.total} answered. `,
      h("span", { class: "row", style: { marginTop: ".5rem" } },
        btn("Resume", { size: "sm", kind: "primary", icon: "play", onClick: () => go("quiz/resume") }),
        btn("Discard", { size: "sm", kind: "ghost", onClick: async (e) => { await del("/session").catch(() => {}); e.currentTarget.closest(".alert").remove(); } })))));
  }
  const p = prefs();
  const quizzes = list.quizzes;
  let chosen = quizzes.find((q) => q.id === preselect) || quizzes[0];
  const sel = h("select", { "aria-label": "Quiz", onchange: (e) => { chosen = quizzes.find((q) => q.id === e.target.value); paintInfo(); } },
    quizzes.map((q) => h("option", { value: q.id, selected: q.id === chosen.id }, `${q.title} (${q.n_questions})`)));
  const info = h("div", { class: "row", style: { marginTop: ".6rem" } });
  const paintInfo = () => {
    clear(info).append(tag(plural(chosen.n_questions, "question"), "primary"),
      ...(chosen.difficulties || "").split(",").filter(Boolean).map(difficultyTag),
      chosen.best_percent != null ? tag(`your best ${Math.round(chosen.best_percent)}%`) : tag("not taken yet"));
    minutes.value = String(p.minutes || chosen.n_questions);
  };
  const minutes = h("input", { type: "number", min: 1, max: 300, value: p.minutes, style: { maxWidth: "100px" }, hidden: p.timer !== "countdown", oninput: (e) => { p.minutes = parseInt(e.target.value || "1", 10); } });
  paintInfo();
  root.append(h("div", { class: "card pad-lg" },
    h("div", { class: "stack" },
      field("Quiz", sel), info,
      h("div", { class: "grid cols-3" },
        field("Layout", seg([{ value: "one", label: "One at a time" }, { value: "all", label: "All on one page" }], p.mode, (v) => { p.mode = v; })),
        field("Feedback", seg([{ value: true, label: "Instant" }, { value: false, label: "Exam mode" }], p.instant, (v) => { p.instant = v === true || v === "true"; }),
          "Instant shows right or wrong after each answer."),
        field("Timer", h("div", { class: "row" }, seg([{ value: "off", label: "Off" }, { value: "stopwatch", label: "Stopwatch" }, { value: "countdown", label: "Countdown" }], p.timer,
          (v) => { p.timer = v; minutes.hidden = v !== "countdown"; }), minutes), "Countdown is in minutes; the quiz is submitted when time runs out.")),
      h("div", { class: "row", style: { marginTop: ".4rem" } },
        btn("Start quiz", { kind: "primary lg", icon: "play", onClick: async () => {
          savePrefs(p);
          try {
            const quiz = (await get(`/quizzes/${chosen.id}`)).quiz;
            begin(quiz, p);
            history.replaceState(null, "", `#/quiz/${quiz.id}`);
            clear(root);
            renderRun(root);
          } catch (err) { toast(err.message, "bad"); }
        } }),
        h("span", { class: "small muted" }, "Shortcuts: ", h("kbd", null, "A"), "–", h("kbd", null, "F"), " answer, ", h("kbd", null, "←"), h("kbd", null, "→"), " move, ", h("kbd", null, "M"), " flag")))));
}

// -------------------------------------------------------------------------------------------------- run
function renderRun(root) {
  const qs = R.quiz.questions;
  const timerEl = h("span", { class: "timer", hidden: R.timer === "off" });
  const bar = h("div", { class: "progress" }, h("span"));
  const left = h("div", { class: "stack" });
  const palette = h("div", { class: "palette" });
  const answeredLine = h("div", { class: "small muted" });

  root.append(h("div", { class: "quiz-head" },
    h("div", null, h("div", { class: "tiny muted" }, R.subset ? "RETRY OF MISSED QUESTIONS" : "QUIZ"), h("h1", null, R.quiz.title)),
    h("span", { class: "spacer" }), timerEl,
    btn(R.subset ? "Exit" : "Save and exit", { kind: "ghost", icon: "chevronLeft", onClick: () => { persist.flush(); R = null; go("quiz"); } })));
  root.append(h("div", { style: { marginBottom: "1rem" } }, bar));
  root.append(h("div", { class: "quiz-wrap" }, left,
    h("aside", { class: "card", style: { position: "sticky", top: "76px" } },
      h("h3", { style: { marginBottom: ".7rem" } }, "Questions"), palette,
      h("div", { class: "legend" }, h("span", null, h("i", { style: { background: "var(--primary-soft)" } }), "answered"),
        h("span", null, h("i", { style: { background: "var(--warn)" } }), "flagged"), h("span", null, h("i", { style: { background: "var(--surface-3)" } }), "open")),
      h("div", { style: { marginTop: "1rem" } }, answeredLine),
      h("div", { style: { marginTop: ".8rem" } }, btn("Submit quiz", { kind: "primary", icon: "check", onClick: () => submit(root) })))));

  const locked = (q) => R.instant && R.answers[q.id] != null;
  function choiceList(q) {
    const sel = R.answers[q.id];
    return h("div", { class: "choices", role: "radiogroup", "aria-label": "Options" }, q.options.map((o) => {
      let cls = "choice";
      if (sel === o.id) cls += " selected";
      if (locked(q)) {
        if (o.id === q.correct_answer) cls += " right";
        else if (sel === o.id) cls += " wrong";
      }
      return h("button", { type: "button", class: cls, role: "radio", "aria-checked": String(sel === o.id), disabled: locked(q),
        onclick: () => pick(q, o.id) }, h("span", { class: "key" }, o.id), h("span", null, o.text));
    }));
  }
  function feedback(q) {
    if (!locked(q)) return null;
    const ok = R.answers[q.id] === q.correct_answer;
    const right = q.options.find((o) => o.id === q.correct_answer);
    return h("div", { class: "feedback" }, alertBox(ok ? "good" : "bad",
      h("b", null, ok ? "Correct!" : `Not quite — the answer is ${right.id}. ${right.text}`), q.explanation ? h("div", { style: { marginTop: ".3rem" } }, q.explanation) : null));
  }
  function questionBlock(q, i) {
    return h("section", { class: "card", dataset: { qid: q.id } },
      h("div", { class: "row between" }, h("div", { class: "row" }, h("span", { class: "q-num" }, `Question ${i + 1} of ${qs.length}`), difficultyTag(q.difficulty),
        q.topic && q.topic !== "General" ? tag(q.topic) : null),
        btn(R.flagged.has(q.id) ? "Flagged" : "Flag", { size: "sm", kind: R.flagged.has(q.id) ? "" : "ghost", icon: "flag", onClick: () => flag(q) })),
      h("div", { class: "quiz-q" }, q.question), choiceList(q), feedback(q));
  }
  function paint() {
    clear(left);
    if (R.mode === "all") {
      qs.forEach((q, i) => left.append(questionBlock(q, i)));
      left.append(h("div", { class: "row end" }, btn("Submit quiz", { kind: "primary lg", icon: "check", onClick: () => submit(root) })));
    } else {
      const i = R.current, q = qs[i];
      left.append(questionBlock(q, i), h("div", { class: "row between" },
        btn("Previous", { icon: "chevronLeft", disabled: i === 0, onClick: () => move(-1) }),
        i < qs.length - 1 ? btn("Next", { kind: "primary", iconRight: "chevronRight", onClick: () => move(1) })
          : btn("Submit quiz", { kind: "primary", icon: "check", onClick: () => submit(root) })));
    }
    paintSide();
  }
  function paintSide() {
    clear(palette);
    qs.forEach((q, i) => {
      let cls = "";
      if (R.answers[q.id] != null) cls += R.instant ? (R.answers[q.id] === q.correct_answer ? " right" : " wrong") : " answered";
      if (R.flagged.has(q.id)) cls += " flagged";
      if (R.mode === "one" && i === R.current) cls += " current";
      palette.append(h("button", { type: "button", class: cls.trim(), "aria-label": `Question ${i + 1}`, onclick: () => jump(i) }, i + 1));
    });
    const n = Object.keys(R.answers).filter((k) => qs.some((q) => q.id === k)).length;
    answeredLine.textContent = `${n} of ${qs.length} answered${R.flagged.size ? ` · ${R.flagged.size} flagged` : ""}`;
    bar.firstChild.style.width = `${(100 * n) / qs.length}%`;
  }
  function pick(q, id) {
    if (locked(q)) return;
    R.answers[q.id] = id;
    persist();
    if (R.mode === "all") {
      const block = left.querySelector(`[data-qid="${q.id}"]`);
      block.replaceWith(questionBlock(q, qs.indexOf(q)));
      paintSide();
    } else paint();
  }
  function flag(q) {
    if (R.flagged.has(q.id)) R.flagged.delete(q.id); else R.flagged.add(q.id);
    persist();
    if (R.mode === "all") { left.querySelector(`[data-qid="${q.id}"]`).replaceWith(questionBlock(q, qs.indexOf(q))); paintSide(); } else paint();
  }
  function move(d) { R.current = Math.max(0, Math.min(qs.length - 1, R.current + d)); persist(); paint(); }
  function jump(i) {
    if (R.mode === "one") { R.current = i; persist(); paint(); }
    else left.querySelector(`[data-qid="${qs[i].id}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  const onKey = (e) => {
    if (e.target.closest("input, textarea, select, dialog") || e.ctrlKey || e.metaKey || e.altKey) return;
    const q = R.mode === "one" ? qs[R.current] : null;
    const k = e.key.toUpperCase();
    if (q && /^[A-H]$/.test(k) && q.options.some((o) => o.id === k)) { pick(q, k); e.preventDefault(); }
    else if (q && /^[1-8]$/.test(k) && q.options[Number(k) - 1]) { pick(q, q.options[Number(k) - 1].id); e.preventDefault(); }
    else if (e.key === "ArrowRight" && R.mode === "one") { move(1); e.preventDefault(); }
    else if (e.key === "ArrowLeft" && R.mode === "one") { move(-1); e.preventDefault(); }
    else if (k === "M" && q) { flag(q); e.preventDefault(); }
  };
  document.addEventListener("keydown", onKey);

  const tick = () => {
    if (!R || R.result || R.timer === "off") return;
    if (R.limit) {
      const left_ = R.limit - elapsed();
      timerEl.textContent = fmtDuration(left_);
      timerEl.classList.toggle("low", left_ <= 30);
      if (left_ <= 0) { clearInterval(timer); toast("Time is up — your quiz was submitted.", "warn"); submit(root, true); }
    } else timerEl.textContent = fmtDuration(elapsed());
  };
  const timer = setInterval(tick, 500);
  tick();
  paint();
  runCleanup?.();
  runCleanup = () => { document.removeEventListener("keydown", onKey); clearInterval(timer); persist.flush(); };
}

async function submit(root, force = false) {
  if (!R || R.result) return;
  const qs = R.quiz.questions;
  const open = qs.filter((q) => R.answers[q.id] == null).length;
  if (!force) {
    const msg = [h("p", null, open ? `${plural(open, "question")} ${open === 1 ? "is" : "are"} still unanswered and will score zero.` : "All questions are answered.")];
    if (R.flagged.size) msg.push(h("p", null, `${plural(R.flagged.size, "question")} ${R.flagged.size === 1 ? "is" : "are"} flagged for review.`));
    const ok = await confirmDialog({ title: "Submit your answers?", message: msg, confirm: "Submit", cancel: "Keep going" });
    if (!ok) return;
  }
  const duration = R.timer === "off" ? null : Math.min(elapsed(), R.limit || Infinity);
  try {
    const res = await post(`/quizzes/${R.quiz.id}/score`, { answers: R.answers, duration_s: duration, question_ids: R.subset });
    R.elapsedBefore = elapsed();
    R.result = res;
    runCleanup?.();
    runCleanup = null;
    clear(root);
    document.dispatchEvent(new Event("quizforge:submitted"));
    renderResults(root);
    window.scrollTo(0, 0);
  } catch (err) { toast(err.message, "bad", 7000); }
}

// ---------------------------------------------------------------------------------------------- results
function renderResults(root) {
  const { attempt, recommendations, by_topic: byTopic, by_difficulty: byDiff, saved } = R.result;
  const qs = R.quiz.questions;
  const detail = Object.fromEntries(attempt.details.map((d) => [d.question_id, d]));
  const missed = qs.filter((q) => !detail[q.id]?.is_correct);
  const pct = attempt.percent;
  const verdict = pct >= 90 ? "Outstanding!" : pct >= 75 ? "Great work!" : pct >= 50 ? "Good effort — keep practising." : "Keep going — review the explanations below.";

  root.append(h("div", { class: "quiz-head" }, h("div", null, h("div", { class: "tiny muted" }, "RESULTS"), h("h1", null, R.quiz.title))));
  root.append(h("section", { class: "card pad-lg", style: { marginBottom: "1rem" } },
    h("div", { class: "result-hero" }, scoreRing(pct),
      h("div", { class: "stack", style: { gap: ".8rem" } },
        h("h2", null, verdict),
        h("div", { class: "result-counts" },
          h("div", null, h("b", { style: { color: "var(--good)" } }, attempt.correct), h("span", null, "correct")),
          h("div", null, h("b", { style: { color: "var(--bad)" } }, attempt.incorrect), h("span", null, "wrong")),
          h("div", null, h("b", { style: { color: "var(--warn)" } }, attempt.unanswered), h("span", null, "skipped")),
          attempt.duration_s != null ? h("div", null, h("b", null, fmtDuration(attempt.duration_s)), h("span", null, "time")) : null),
        h("div", { class: "row" },
          btn("Retake quiz", { icon: "rotate", onClick: () => { const quiz = R.quiz; const opts = { mode: R.mode, instant: R.instant, timer: R.timer, minutes: (R.limit || 600) / 60, subset: R.subset }; begin(quiz, opts); clear(root); renderRun(root); } }),
          missed.length && missed.length < qs.length ? btn(`Retry ${plural(missed.length, "missed question")}`, { kind: "primary", icon: "target", onClick: async () => {
            const full = (await get(`/quizzes/${R.quiz.id}`)).quiz;
            begin(full, { mode: R.mode, instant: true, timer: "off", subset: missed.map((q) => q.id) });
            clear(root); renderRun(root);
          } }) : null,
          saved ? menu(btn("Export results", { icon: "download" }), [
            { label: "Results (CSV)", icon: "download", onClick: () => download(`/attempts/${attempt.id}/export/csv`).catch((e) => toast(e.message, "bad")) },
            { label: "Results (JSON)", icon: "download", onClick: () => download(`/attempts/${attempt.id}/export/json`).catch((e) => toast(e.message, "bad")) },
          ]) : h("span", { class: "small muted" }, "Practice round — not added to your history."),
          btn("Other quizzes", { kind: "ghost", onClick: () => { R = null; go("quiz"); } }))))));

  const recCard = h("section", { class: "card" }, h("h2", { style: { marginBottom: ".7rem" } }, "What to study next"),
    h("ul", { class: "stack", style: { gap: ".4rem", paddingLeft: "1.1rem", margin: 0 } }, recommendations.map((r) => h("li", null, r))));
  const perf = h("section", { class: "card" }, h("h2", { style: { marginBottom: ".7rem" } }, "By topic and difficulty"));
  if (byTopic.length) perf.append(accuracyBars(byTopic.map((r) => ({ label: r.topic, value: r.accuracy, detail: `${r.correct}/${r.total}` })), { title: "Accuracy by topic" }));
  if (byDiff.length) perf.append(h("div", { style: { marginTop: ".8rem" } }, accuracyBars(byDiff.map((r) => ({ label: r.difficulty, value: r.accuracy, detail: `${r.correct}/${r.total}` })), { title: "Accuracy by difficulty" })));
  root.append(h("div", { class: "grid cols-2", style: { marginBottom: "1rem" } }, recCard, perf));

  const list = h("div", { class: "q-list" });
  const paintReview = () => {
    clear(list);
    const pickQs = qs.filter((q) => {
      const d = detail[q.id];
      if (R.review === "wrong") return d && !d.is_correct && d.selected != null;
      if (R.review === "skipped") return !d || d.selected == null;
      if (R.review === "flagged") return R.flagged.has(q.id);
      return true;
    });
    if (!pickQs.length) list.append(h("p", { class: "muted" }, "Nothing in this view."));
    for (const q of pickQs) {
      const d = detail[q.id];
      const state = d?.is_correct ? "ok" : d?.selected == null ? "skip" : "no";
      list.append(h("article", { class: `q-card review-item ${state}` },
        h("div", { class: "q-top" }, h("span", { class: "q-num" }, `Q${qs.indexOf(q) + 1}`),
          tag(state === "ok" ? "Correct" : state === "skip" ? "Skipped" : "Wrong", state === "ok" ? "good" : state === "skip" ? "medium" : "bad"),
          difficultyTag(q.difficulty), R.flagged.has(q.id) ? tag("Flagged") : null),
        h("div", { class: "q-text" }, q.question),
        h("ul", { class: "q-opts" }, q.options.map((o) => h("li", { class: o.id === q.correct_answer ? "correct" : d?.selected === o.id ? "wrong-pick" : "" },
          h("span", { class: "letter" }, o.id), h("span", null, o.text), h("span", { class: "spacer" }),
          o.id === q.correct_answer ? icon("check") : d?.selected === o.id ? h("span", { class: "tag bad" }, "your answer") : null))),
        q.explanation || q.source_excerpt ? h("div", { class: "q-extra" },
          q.explanation ? h("div", null, h("span", { class: "label" }, "Explanation"), q.explanation) : null,
          q.source_excerpt ? h("div", { class: "q-source" }, h("span", { class: "label" }, "From your material"), `“${q.source_excerpt}”`) : null) : null));
    }
  };
  R.review = R.review || "all";
  root.append(h("section", null,
    h("div", { class: "row between", style: { margin: ".4rem 0 .8rem" } }, h("h2", null, "Review your answers"),
      seg([{ value: "all", label: "All" }, { value: "wrong", label: "Wrong" }, { value: "skipped", label: "Skipped" }, { value: "flagged", label: "Flagged" }], R.review,
        (v) => { R.review = v; paintReview(); })), list));
  paintReview();
}
