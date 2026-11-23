import { api, del, download, get, pollJob, post, put } from "../api.js";
import { go, handOff } from "../nav.js";
import { QuestionEditor, questionCard } from "../questions.js";
import { loadSettings, onStatus, refreshStatus, store } from "../store.js";
import {
  alertBox, btn, busy, clear, confirmDialog, debounce, field, h, icon, menu, pageHead, plural, seg, stepper, toast, toggle,
} from "../ui.js";

// Module state survives page changes, so a running generation keeps going while you look elsewhere.
const CFG_KEY = "quizforge-generate-settings";
const MATERIAL_KEY = "quizforge-material";
const kept = (() => { try { return JSON.parse(sessionStorage.getItem(MATERIAL_KEY) || "{}"); } catch { return {}; } })();
const S = {
  material: kept.material || "",
  sourceName: kept.sourceName || "",
  fileNote: null, // {name, pages, words, warnings}
  cfg: loadCfg(),
  check: null, // result of /material/check
  job: null, // {id, append, request, questions[], snap, kind}
  draft: undefined, // undefined = not loaded yet, null = none
  savedNotice: null,
  paint: null, // whole page (progress + review), set while the page is open
  paintRun: null, // progress panel only (called on every poll)
};

function loadCfg() {
  const base = { topic: "", count: 10, custom: false, difficulty: "Medium", style: "Mixed", num_options: 4, include_explanations: true,
    include_sources: true, instructions: "", double_check: false };
  try { return { ...base, ...JSON.parse(localStorage.getItem(CFG_KEY) || "{}"), topic: "", instructions: "" }; } catch { return base; }
}
function saveCfg() {
  try { localStorage.setItem(CFG_KEY, JSON.stringify(S.cfg)); } catch { /* not remembered */ }
}

function request(count) {
  const c = S.cfg;
  return { topic: c.topic.trim(), count: count ?? c.count, difficulty: c.difficulty, style: c.style, num_options: c.num_options,
    instructions: c.instructions.trim(), include_explanations: c.include_explanations, include_sources: c.include_sources };
}

// the text box survives a page reload in this browser tab (not on the server: it may be private notes)
const keepMaterial = debounce(() => {
  try { sessionStorage.setItem(MATERIAL_KEY, JSON.stringify({ material: S.material, sourceName: S.sourceName })); } catch { /* too large or blocked */ }
}, 800);

const persistDraft = debounce(() => {
  if (S.draft) put("/draft", { draft: S.draft }).catch(() => {});
}, 600);

export async function render(root) {
  const settings = await loadSettings().catch(() => null);
  if (S.draft === undefined) {
    try { S.draft = (await get("/draft")).draft || null; } catch { S.draft = null; }
  }
  const limit = settings?.max_input_chars ?? 60000;
  const limits = settings?.limits || { max_upload_mb: 10, styles: ["Conceptual", "Factual", "Application-based", "Mixed"], max_questions: 50, count_presets: [5, 10, 15, 20] };

  // created first: the status callback below repaints the Generate button straight away
  const c = S.cfg;
  const summary = h("div", { class: "small muted" });
  const genBtn = btn("Generate questions", { kind: "primary lg", icon: "sparkles", onClick: () => start(false) });
  const reason = h("div", { class: "small muted" });

  root.append(pageHead("Generate Quiz", "Add your study material, choose the settings, and your local model writes the questions. Each one is checked against the material before you see it."));
  const banner = h("div");
  root.append(banner);
  const unsub = onStatus((st) => {
    banner.replaceChildren();
    if (!st.ready && !st.server_down) {
      banner.append(h("div", { style: { marginBottom: "1rem" } }, alertBox("warn", st.error || "The local model is not ready.",
        " You can prepare your material now; generating unlocks once the model is available. ", h("a", { href: "#/settings" }, "Open Settings"))));
    }
    paintGenerateButton();
  });
  if (store.status?.ready) post("/warmup").catch(() => {});

  // ----------------------------------------------------------------------------------------- 1. material
  const fileInput = h("input", { type: "file", accept: (limits.extensions || [".pdf", ".docx", ".txt"]).join(","), hidden: true });
  const fileNote = h("div");
  const drop = h("div", { class: "dropzone", tabindex: "0", role: "button", "aria-label": "Upload a file" },
    icon("upload"), h("div", null, h("strong", null, "Drop a PDF, DOCX or TXT file here"), " or click to browse"),
    h("div", { class: "small muted" }, `Up to ${limits.max_upload_mb} MB. The file is read on this computer only.`));
  const textarea = h("textarea", { class: "material", placeholder: "...or paste lecture notes, an article or a textbook section here.", value: S.material, "aria-label": "Learning material" });
  const counts = h("div", { class: "small muted" });
  const materialAlerts = h("div", { class: "stack", style: { gap: ".5rem" } });

  const upload = async (file) => {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    drop.classList.add("over");
    clear(fileNote).append(h("div", { class: "row small muted" }, h("span", { class: "pulse-dot" }), `Reading ${file.name}...`));
    try {
      const res = await api("/extract", { method: "POST", form });
      S.material = res.text;
      S.sourceName = res.name;
      S.fileNote = res;
      textarea.value = res.text;
      if (!S.cfg.topic) { S.cfg.topic = res.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").slice(0, 60); topicInput.value = S.cfg.topic; }
      paintFileNote();
      checkMaterial.flush();
      toast(`Read ${res.name}: ${plural(res.words, "word")}.`, "good");
    } catch (err) {
      clear(fileNote).append(alertBox("bad", err.message));
    } finally {
      drop.classList.remove("over");
      fileInput.value = "";
    }
  };
  const paintFileNote = () => {
    clear(fileNote);
    if (!S.fileNote) return;
    fileNote.append(h("div", { class: "row" },
      h("span", { class: "file-chip" }, icon("file"), S.fileNote.name),
      h("span", { class: "small muted" }, `${S.fileNote.pages ? plural(S.fileNote.pages, "page") + " · " : ""}${plural(S.fileNote.words, "word")} extracted — review the text below`),
      btn("Remove", { size: "sm", kind: "ghost", icon: "x", onClick: () => { S.fileNote = null; S.sourceName = ""; S.material = ""; textarea.value = ""; paintFileNote(); checkMaterial.flush(); } })));
    for (const w of S.fileNote.warnings || []) fileNote.append(h("div", { style: { marginTop: ".5rem" } }, alertBox("info", w)));
  };
  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); upload(e.dataTransfer.files[0]); });
  fileInput.addEventListener("change", () => upload(fileInput.files[0]));

  const paintCounts = () => {
    keepMaterial();
    const text = S.material;
    const c = S.check;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    counts.textContent = `${text.length.toLocaleString()} characters · ${words.toLocaleString()} words` +
      (c && c.sections ? ` · ${plural(c.sections, "section")}` : "") + ` · limit ${limit.toLocaleString()} characters`;
    clear(materialAlerts);
    if (text.length > limit) {
      materialAlerts.append(alertBox("bad", `The material is ${(text.length - limit).toLocaleString()} characters over the limit. Nothing is cut silently: shorten it, or keep only the first part. `,
        btn(`Keep the first ${limit.toLocaleString()} characters`, { size: "sm", onClick: () => { S.material = S.material.slice(0, limit); textarea.value = S.material; checkMaterial.flush(); } })));
    }
    if (c?.refused) {
      materialAlerts.append(alertBox("bad", "This text only gives instructions to an AI and has nothing to write questions about. QuizForge will refuse it."));
    } else if (c?.removed?.length) {
      materialAlerts.append(alertBox("warn", h("b", null, `${plural(c.removed.length, "sentence")} in the material ${c.removed.length === 1 ? "gives" : "give"} instructions to the AI and will be ignored`),
        h("ul", null, c.removed.slice(0, 3).map((s) => h("li", { class: "small" }, s.length > 140 ? s.slice(0, 139) + "…" : s)))));
    }
    paintGenerateButton();
  };
  const checkMaterial = debounce(async () => {
    if (!S.material.trim()) { S.check = null; paintCounts(); return; }
    try { S.check = await post("/material/check", { text: S.material }); } catch { S.check = null; }
    paintCounts();
  }, 500);
  textarea.addEventListener("input", () => { S.material = textarea.value; paintCounts(); checkMaterial(); });

  root.append(h("section", { class: "card", style: { marginBottom: "1rem" } },
    h("div", { class: "card-head" }, h("span", { class: "step" }, "1"), h("h2", null, "Learning material")),
    h("div", { class: "stack" }, drop, fileInput, fileNote, textarea, counts, materialAlerts)));
  paintFileNote();
  paintCounts();
  if (S.material && !S.check) checkMaterial.flush();

  // ------------------------------------------------------------------------------------------ 2. settings
  const topicInput = h("input", { type: "text", value: c.topic, placeholder: "e.g. Photosynthesis", oninput: (e) => { c.topic = e.target.value; paintSummary(); } });
  const customCount = h("input", { type: "number", min: 1, max: limits.max_questions, value: c.count, hidden: !c.custom, style: { maxWidth: "110px" },
    oninput: (e) => { c.count = Math.max(1, Math.min(limits.max_questions, parseInt(e.target.value || "1", 10))); saveCfg(); paintSummary(); } });
  const countSeg = seg([...limits.count_presets.map((n) => ({ value: n, label: String(n) })), { value: "custom", label: "Custom" }],
    c.custom ? "custom" : c.count, (v) => {
      c.custom = v === "custom";
      customCount.hidden = !c.custom;
      if (!c.custom) c.count = v; else customCount.focus();
      saveCfg(); paintSummary();
    }, { label: "Number of questions" });
  const styleSel = h("select", { onchange: (e) => { c.style = e.target.value; saveCfg(); paintSummary(); } },
    limits.styles.map((s) => h("option", { value: s, selected: s === c.style }, s === "Application-based" ? "Application" : s)));
  const modelPick = h("div", { class: "model-pick" });
  const paintModels = (st) => {
    clear(modelPick);
    const models = (st?.models || []).filter((m) => m.installed);
    if (!models.length) { modelPick.append(h("span", { class: "small muted" }, "No model installed yet — see Settings.")); return; }
    for (const m of models) {
      modelPick.append(h("button", { type: "button", "aria-pressed": String(!!m.active), disabled: !!S.job, onclick: async () => {
        if (m.active) return;
        try { await post("/model", { tag: m.tag }); await refreshStatus(true); toast(`Now using the ${m.tier} model.`, "good"); }
        catch (err) { toast(err.message, "bad"); }
      } }, h("b", null, m.tier), h("small", null, m.finetuned ? "fine-tuned for QuizForge" : m.tier === "Fast" ? "base model, quicker" : "base model, more accurate")));
    }
  };
  const unsubModels = onStatus(paintModels);

  root.append(h("section", { class: "card", style: { marginBottom: "1rem" } },
    h("div", { class: "card-head" }, h("span", { class: "step" }, "2"), h("h2", null, "Quiz settings")),
    h("div", { class: "stack" },
      h("div", { class: "grid cols-2" },
        field("Topic or subject", topicInput, "Used for the quiz title and to label questions."),
        field("Number of questions", h("div", { class: "row" }, countSeg, customCount))),
      h("div", { class: "grid fit" },
        field("Difficulty", seg(["Easy", "Medium", "Hard", "Mixed"], c.difficulty, (v) => { c.difficulty = v; saveCfg(); paintSummary(); }, { label: "Difficulty" })),
        field("Question style", styleSel),
        field("Options per question", stepper(c.num_options, 2, 6, (v) => { c.num_options = v; saveCfg(); paintSummary(); }))),
      h("div", { class: "grid fit" },
        toggle("Explanations", c.include_explanations, (v) => { c.include_explanations = v; saveCfg(); }, "Why the answer is right"),
        toggle("Source quotes", c.include_sources, (v) => { c.include_sources = v; saveCfg(); }, "Show the supporting sentence"),
        toggle("Double-check answer keys", c.double_check, (v) => { c.double_check = v; saveCfg(); }, "Slower: the model re-answers blind")),
      field("Model", modelPick, "Fine-tuned models were trained for exactly this task. Manage models in Settings."),
      field("Extra instructions (optional)", h("textarea", { rows: 2, value: c.instructions, placeholder: "e.g. Focus on definitions. Avoid questions about dates.", oninput: (e) => { c.instructions = e.target.value; } })))));

  // ------------------------------------------------------------------------------------------ 3. generate
  const runSlot = h("div");
  const liveList = h("div", { class: "q-list", style: { marginTop: "1rem" } });
  function paintSummary() {
    summary.textContent = `${plural(c.count, "question")} · ${c.difficulty} · ${c.style === "Application-based" ? "Application" : c.style} · ${c.num_options} options each`;
    clear(genBtn).append(icon("sparkles"), `Generate ${plural(c.count, "question")}`);
  }
  function paintGenerateButton() {
    const st = store.status;
    const why = S.job ? "A generation is already running." : !st?.ready ? "Start Ollama and install the model first."
      : S.material.length > limit ? "Shorten the material first." : S.check?.refused ? "This material cannot be used."
      : !S.material.trim() && !c.topic.trim() ? "Add material, or at least a topic." : "";
    genBtn.disabled = !!why;
    reason.textContent = why;
  }
  root.append(h("section", { class: "card", style: { marginBottom: "1rem" } },
    h("div", { class: "card-head" }, h("span", { class: "step" }, "3"), h("h2", null, "Generate")),
    h("div", { class: "row between" }, h("div", null, summary, reason), genBtn), runSlot, liveList));
  paintSummary();
  paintGenerateButton();

  // ---------------------------------------------------------------------------------------- 4. review
  const reviewSlot = h("div");
  root.append(reviewSlot);

  function paintRun() {
    clear(runSlot);
    paintGenerateButton();
    paintModels(store.status);
    if (!S.job) return;
    const snap = S.job.snap;
    const got = S.job.questions.length;
    const want = S.job.request.count;
    const pct = snap?.progress?.message?.startsWith("Double-checking") && snap.progress.total
      ? (100 * snap.progress.done) / snap.progress.total : (100 * got) / Math.max(1, want);
    const bar = h("div", { class: `progress ${got === 0 ? "indeterminate" : ""}` }, h("span"));
    if (got) bar.firstChild.style.width = `${Math.min(100, pct)}%`;
    runSlot.append(h("div", { class: "card gen-panel", style: { marginTop: "1rem" } },
      h("div", { class: "row between" },
        h("div", { class: "row" }, h("span", { class: "pulse-dot" }), h("b", null, S.job.kind === "check" ? "Double-checking answer keys..." : `Writing questions with the ${store.status?.model_name || "local"} model...`)),
        btn("Stop", { icon: "stop", kind: "danger", onClick: async (e) => {
          await busy(e.currentTarget, () => post(`/jobs/${S.job.id}/stop`).catch(() => {}));
        } })),
      h("div", { style: { marginTop: ".8rem" } }, bar),
      h("div", { class: "gen-meta" },
        S.job.kind === "check" ? null : h("span", null, h("b", null, `${got} of ${want}`), " ready"),
        h("span", null, snap?.progress?.message || "Starting the model..."),
        h("span", null, `${Math.round(snap?.elapsed_s || 0)} s`)),
      S.job.kind === "check" ? null : h("div", { class: "tiny muted", style: { marginTop: ".4rem" } }, "Each question appears as soon as it passes validation. Stop keeps what is already written.")));
  }

  function paintReview() {
    clear(reviewSlot);
    if (S.savedNotice) {
      const n = S.savedNotice;
      reviewSlot.append(h("section", { class: "card", style: { marginBottom: "1rem" } },
        alertBox("good", h("b", null, `Saved “${n.title}” to your question bank`), ` (${plural(n.count, "question")}).`),
        h("div", { class: "row", style: { marginTop: ".8rem" } },
          btn("Take this quiz", { kind: "primary", icon: "play", onClick: () => go(`quiz/${n.id}`) }),
          btn("Open in bank", { icon: "library", onClick: () => go(`bank/${n.id}`) }),
          btn("Generate another", { kind: "ghost", icon: "sparkles", onClick: () => { S.savedNotice = null; paintReview(); window.scrollTo({ top: 0, behavior: "smooth" }); } }))));
    }
    const d = S.draft;
    if (!d) return;
    const quiz = d.quiz;
    const card = h("section", { class: "card", style: { marginBottom: "1rem" } });
    card.append(h("div", { class: "card-head" }, h("span", { class: "step" }, "4"), h("h2", null, "Review and save"),
      h("span", { class: "sub" }, `${plural(quiz.questions.length, "question")}${d.target && d.target > quiz.questions.length ? ` of ${d.target}` : ""}`)));
    const alerts = h("div", { class: "stack", style: { gap: ".5rem", marginBottom: "1rem" } });
    for (const w of d.warnings || []) alerts.append(alertBox(w.startsWith("Ignored") ? "info" : "warn", w));
    if (d.rejected && !(d.warnings || []).some((w) => w.includes("rejected"))) alerts.append(h("div", { class: "small muted" }, `${plural(d.rejected, "candidate")} written by the model failed validation and ${d.rejected === 1 ? "is" : "are"} not shown.`));
    const need = (d.target || 0) - quiz.questions.length;
    if (need > 0 && !S.job) {
      alerts.append(h("div", null, btn(`Generate the remaining ${plural(need, "question")}`, { icon: "plus", disabled: !store.status?.ready, onClick: () => start(true, need) })));
    }
    if (alerts.childNodes.length) card.append(alerts);

    const title = h("input", { type: "text", value: quiz.title, oninput: (e) => { quiz.title = e.target.value; persistDraft(); } });
    const subject = h("input", { type: "text", value: quiz.subject || "", oninput: (e) => { quiz.subject = e.target.value; persistDraft(); } });
    card.append(h("div", { class: "grid cols-2", style: { marginBottom: "1rem" } }, field("Quiz title", title), field("Subject", subject)));

    const editorHost = h("div");
    let editor;
    const revealToggle = toggle("Show all answers", false, (v) => editor.setRevealAll(v));
    const unchecked = quiz.questions.filter((q) => q.key_check === "unchecked").length;
    card.append(h("div", { class: "row between", style: { marginBottom: ".8rem" } },
      h("div", { class: "small muted" }, "Answer keys are written by the AI model. Please check them before using the quiz."),
      h("div", { class: "row" }, revealToggle,
        unchecked && !S.job ? btn("Double-check keys", { size: "sm", icon: "shield", disabled: !store.status?.ready, onClick: () => startCheck() }) : null,
        exportMenu(() => d.quiz))));
    editor = new QuestionEditor(editorHost, { questions: quiz.questions, sourceText: quiz.source_text, request: d.request },
      { onChange: () => { persistDraft(); paintSaveBar(); }, showSource: d.request?.include_sources !== false, numOptions: d.request?.num_options || 4,
        canRegenerate: () => !!store.status?.ready && !S.job });
    card.append(editorHost);

    const saveBar = h("div", { class: "row", style: { marginTop: "1.2rem", paddingTop: "1rem", borderTop: "1px solid var(--border)" } });
    function paintSaveBar() {
      clear(saveBar);
      const none = !quiz.questions.length;
      saveBar.append(
        btn("Save to question bank", { kind: "primary", icon: "library", disabled: none || !!S.job, onClick: (e) => save(e.currentTarget, false) }),
        btn("Save and take quiz", { icon: "play", disabled: none || !!S.job, onClick: (e) => save(e.currentTarget, true) }),
        h("span", { class: "spacer" }),
        btn("Discard draft", { kind: "ghost danger", icon: "trash", disabled: !!S.job, onClick: discard }));
    }
    paintSaveBar();
    card.append(saveBar);
    reviewSlot.append(card);
  }

  async function save(button, take) {
    await busy(button, async () => {
      try {
        const res = await post("/quizzes", { quiz: S.draft.quiz });
        await del("/draft").catch(() => {});
        S.savedNotice = { id: res.quiz.id, title: res.quiz.title, count: res.quiz.questions.length };
        S.draft = null;
        if (res.warning) toast(res.warning, "warn", 6000);
        if (take) { handOff("start-quiz", res.quiz.id); go(`quiz/${res.quiz.id}`); return; }
        toast("Saved to your question bank.", "good");
        paintReview();
        reviewSlot.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (err) {
        toast(err.message, "bad", 7000);
      }
    });
  }

  async function discard() {
    const ok = await confirmDialog({ title: "Discard this draft?", message: "The generated questions that are not saved will be lost.", confirm: "Discard", danger: true });
    if (!ok) return;
    S.draft = null;
    await del("/draft").catch(() => {});
    paintReview();
  }

  S.paint = () => { paintRun(); paintReview(); };
  S.paintRun = paintRun;
  paintRun();
  paintReview();
  if (S.job) for (const q of S.job.questions) liveList.append(questionCard(q, liveList.childElementCount + 1, { showSource: false }));
  S.onQuestion = (q) => liveList.append(questionCard(q, liveList.childElementCount + 1 + (S.job?.append ? (S.draft?.quiz?.questions?.length || 0) : 0), { showSource: false, enter: true }));
  S.clearLive = () => clear(liveList);

  return () => { S.paint = null; S.paintRun = null; S.onQuestion = null; S.clearLive = null; unsub(); unsubModels(); persistDraft.flush(); };
}

function exportMenu(getQuiz) {
  const item = (label, kind) => ({ label, icon: "download", onClick: async () => {
    try { await download(`/export/${kind}`, { quiz: getQuiz() }); } catch (err) { toast(err.message, "bad"); }
  } });
  return menu(btn("Export", { size: "sm", icon: "download" }), [
    { heading: "For students" }, item("Question paper (PDF)", "paper.pdf"), item("Question paper (HTML)", "paper.html"),
    { heading: "For teachers" }, item("Answer key (PDF)", "key.pdf"), item("Answer key (CSV)", "key.csv"),
    "-", item("All questions (JSON)", "questions.json"), item("All questions (CSV)", "questions.csv"),
  ]);
}

// --------------------------------------------------------------------------------------- generation job
async function start(append, need) {
  const req = request(append ? need : undefined);
  const base = append ? (S.draft?.request || req) : req;
  const payload = {
    material: append ? S.draft.quiz.source_text : S.material,
    request: append ? { ...base, count: need } : req,
    existing: append ? S.draft.quiz.questions : [],
    double_check: !append && S.cfg.double_check,
  };
  let res;
  try {
    res = await post("/generate", payload);
  } catch (err) {
    toast(err.message, "bad", 7000);
    if (err.status === 409) refreshStatus(true);
    return;
  }
  if (!append) S.savedNotice = null;
  S.job = { id: res.job_id, kind: "generate", append, request: payload.request, questions: [], snap: null, material: payload.material, sourceName: S.sourceName };
  S.clearLive?.();
  S.paint?.();
  const job = S.job;
  const poll = pollJob(job.id, (snap) => {
    job.snap = snap;
    for (const q of snap.new_questions) { job.questions.push(q); S.onQuestion?.(q); }
    S.paintRun?.();
  });
  let final;
  try { final = await poll.done; } catch (err) { final = { status: "error", error: err.message, warnings: [], questions: job.questions, rejected: 0 }; }
  finishGeneration(job, final);
}

function finishGeneration(job, snap) {
  const questions = snap.questions || job.questions;
  S.job = null;
  S.clearLive?.();
  const warnings = [...(snap.warnings || [])];
  if (snap.status === "stopped") warnings.push(`Generation was stopped — ${plural(questions.length, "question")} kept.`);
  if (!questions.length) {
    S.paint?.();
    toast(snap.error || (snap.status === "stopped" ? "Stopped before any question was ready." : "No valid questions were produced. Try again, or use a larger model."), "bad", 9000);
    return;
  }
  if (job.append && S.draft) {
    S.draft.quiz.questions.push(...questions);
    S.draft.warnings = warnings;
    S.draft.rejected = (S.draft.rejected || 0) + (snap.rejected || 0);
  } else {
    const req = job.request;
    const name = job.material.trim() ? (job.sourceName || "Pasted text") : "Topic only";
    S.draft = {
      quiz: {
        title: `${req.topic || name.replace(/\.[^.]+$/, "")} quiz`, subject: req.topic || "", questions,
        source_name: name, source_text: job.material, model: store.status?.model || "",
      },
      request: req, target: req.count, warnings, rejected: snap.rejected || 0,
    };
  }
  persistDraft.flush();
  S.paint?.();
  toast(snap.status === "done" ? `${plural(questions.length, "question")} ready for review.` : `${plural(questions.length, "question")} kept.`, "good");
  if (!S.paint) toast("Your generated questions are waiting on the Generate page.", "", 6000);
}

async function startCheck() {
  const quiz = S.draft.quiz;
  let res;
  try { res = await post("/check-keys", { questions: quiz.questions }); } catch (err) { toast(err.message, "bad"); return; }
  S.job = { id: res.job_id, kind: "check", append: false, request: { count: quiz.questions.length }, questions: [], snap: null };
  S.paint?.();
  const job = S.job;
  let final;
  try { final = await pollJob(job.id, (snap) => { job.snap = snap; S.paintRun?.(); }).done; }
  catch (err) { final = { status: "error", error: err.message }; }
  S.job = null;
  if (final.questions?.length === quiz.questions.length) {
    quiz.questions.splice(0, quiz.questions.length, ...final.questions);
    persistDraft.flush();
  }
  if (final.status === "error") toast(final.error, "bad", 7000);
  else toast(final.progress?.message || "Double-check finished.", final.questions?.some((q) => q.key_check === "disagrees") ? "warn" : "good", 6000);
  S.paint?.();
}
