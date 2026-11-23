import { del, download, get, patch, post, put } from "../api.js";
import { go } from "../nav.js";
import { QuestionEditor } from "../questions.js";
import { store } from "../store.js";
import {
  alertBox, btn, busy, clear, confirmDialog, debounce, difficultyTag, emptyState, field, fmtDate, h, menu, pageHead, plural,
  promptDialog, tag, toast, toggle,
} from "../ui.js";

export async function render(root, params) {
  if (params[0]) return renderEditor(root, params[0]);
  root.append(pageHead("Question Bank", "Every quiz you saved, ready to take, edit or export.",
    btn("New quiz", { kind: "primary", icon: "sparkles", onClick: () => go("generate") })));

  const filters = { search: "", subject: "", difficulty: "" };
  const search = h("input", { type: "search", placeholder: "Search titles, questions and topics...", "aria-label": "Search" });
  const subjectSel = h("select", { "aria-label": "Subject" }, h("option", { value: "" }, "All subjects"));
  const diffSel = h("select", { "aria-label": "Difficulty" }, h("option", { value: "" }, "Any difficulty"), ["Easy", "Medium", "Hard"].map((d) => h("option", { value: d }, d)));
  const grid = h("div", { class: "grid auto" });
  const countLine = h("div", { class: "small muted" });
  root.append(h("div", { class: "card", style: { marginBottom: "1rem" } },
    h("div", { class: "grid cols-3" }, search, subjectSel, diffSel)), countLine, h("div", { style: { height: ".6rem" } }), grid);

  let first = true;
  const load = async () => {
    const qs = new URLSearchParams(filters).toString();
    let data;
    try { data = await get(`/quizzes?${qs}`); } catch (err) { grid.replaceChildren(alertBox("bad", err.message)); return; }
    if (first) {
      first = false;
      for (const s of data.subjects) subjectSel.append(h("option", { value: s }, s));
      if (!data.quizzes.length) {
        root.replaceChildren(pageHead("Question Bank", "Every quiz you saved, ready to take, edit or export."),
          emptyState("library", "No quizzes yet", "Generate questions from your notes, review them and save them here.",
            btn("Generate a quiz", { kind: "primary", icon: "sparkles", onClick: () => go("generate") })));
        return;
      }
    }
    countLine.textContent = `${plural(data.quizzes.length, "quiz", "quizzes")}${filters.search || filters.subject || filters.difficulty ? " match your filters" : ""}`;
    grid.replaceChildren(...(data.quizzes.length ? data.quizzes.map((q) => quizCard(q, load)) : [h("p", { class: "muted" }, "Nothing matches these filters.")]));
  };
  const reload = debounce(load, 250);
  search.addEventListener("input", () => { filters.search = search.value; reload(); });
  subjectSel.addEventListener("change", () => { filters.subject = subjectSel.value; load(); });
  diffSel.addEventListener("change", () => { filters.difficulty = diffSel.value; load(); });
  grid.append(h("div", { class: "skeleton", style: { height: "160px" } }));
  await load();
}

function quizCard(q, reload) {
  const diffs = (q.difficulties || "").split(",").filter(Boolean);
  return h("article", { class: "card quiz-card" },
    h("div", { class: "row between", style: { alignItems: "flex-start", flexWrap: "nowrap" } },
      h("h3", null, q.title),
      menu(btn("", { size: "sm", kind: "ghost", icon: "dots", title: "More actions" }), [
        { label: "Rename", icon: "edit", onClick: async () => {
          const title = await promptDialog({ title: "Rename quiz", label: "Title", value: q.title });
          if (!title) return;
          try { await patch(`/quizzes/${q.id}`, { title }); toast("Renamed.", "good"); reload(); } catch (err) { toast(err.message, "bad"); }
        } },
        { label: "Duplicate", icon: "copy", onClick: async () => {
          try { await post(`/quizzes/${q.id}/duplicate`); toast("Copy created.", "good"); reload(); } catch (err) { toast(err.message, "bad"); }
        } },
        "-",
        { heading: "Export" },
        ...exportItems(q.id),
        "-",
        { label: "Delete quiz", icon: "trash", onClick: async () => {
          const ok = await confirmDialog({ title: `Delete “${q.title}”?`, message: "The quiz and all its attempts are deleted permanently.", confirm: "Delete", danger: true });
          if (!ok) return;
          try { await del(`/quizzes/${q.id}`); toast("Quiz deleted.", "good"); reload(); } catch (err) { toast(err.message, "bad"); }
        } },
      ])),
    h("div", { class: "meta" }, q.subject ? tag(q.subject, "primary") : null, ...diffs.map(difficultyTag)),
    h("div", { class: "numbers" },
      h("span", null, h("b", null, q.n_questions), q.n_questions === 1 ? " question" : " questions"),
      h("span", null, h("b", null, q.n_attempts), q.n_attempts === 1 ? " attempt" : " attempts"),
      q.best_percent != null ? h("span", null, "best ", h("b", null, `${Math.round(q.best_percent)}%`)) : null),
    h("div", { class: "tiny muted" }, `Updated ${fmtDate(q.updated_at)}${q.model_name ? ` · written by the ${q.model_name} model` : ""}`),
    h("div", { class: "actions" },
      btn("Take quiz", { kind: "primary", size: "sm", icon: "play", onClick: () => go(`quiz/${q.id}`) }),
      btn("Edit", { size: "sm", icon: "edit", onClick: () => go(`bank/${q.id}`) })));
}

export function exportItems(id) {
  const item = (label, kind) => ({ label, icon: "download", onClick: async () => {
    try { await download(`/quizzes/${id}/export/${kind}`); } catch (err) { toast(err.message, "bad"); }
  } });
  return [item("Question paper (PDF)", "paper.pdf"), item("Question paper (HTML)", "paper.html"), item("Answer key (PDF)", "key.pdf"),
    item("Answer key (CSV)", "key.csv"), item("Questions + answers (JSON)", "questions.json"), item("Questions + answers (CSV)", "questions.csv"),
    item("Questions only (JSON)", "blank.json"), item("Questions only (CSV)", "blank.csv")];
}

async function renderEditor(root, id) {
  let quiz;
  try { quiz = (await get(`/quizzes/${id}`)).quiz; } catch (err) {
    root.append(emptyState("alert", "Quiz not found", err.message, btn("Back to the bank", { onClick: () => go("bank") })));
    return;
  }
  let dirty = false;
  const saveBtn = btn("Save changes", { kind: "primary", icon: "check", disabled: true, onClick: (e) => save(e.currentTarget) });
  const markDirty = () => { dirty = true; saveBtn.disabled = false; };
  root.append(h("div", null, btn("Question bank", { kind: "ghost", size: "sm", icon: "chevronLeft", onClick: () => go("bank") })));
  root.append(pageHead("Edit quiz", "Change wording, fix answer keys, rewrite single questions with the model, or add your own.",
    btn("Take quiz", { icon: "play", onClick: () => go(`quiz/${id}`) }),
    menu(btn("Export", { icon: "download" }), exportItems(id)), saveBtn));

  const title = h("input", { type: "text", value: quiz.title, oninput: (e) => { quiz.title = e.target.value; markDirty(); } });
  const subject = h("input", { type: "text", value: quiz.subject || "", oninput: (e) => { quiz.subject = e.target.value; markDirty(); } });
  const host = h("div");
  let editor;
  root.append(h("div", { class: "card", style: { marginBottom: "1rem" } },
    h("div", { class: "grid cols-2" }, field("Title", title), field("Subject", subject)),
    h("div", { class: "row between", style: { marginTop: "1rem" } },
      h("div", { class: "small muted" }, `${plural(quiz.questions.length, "question")} · created ${fmtDate(quiz.created_at)}`),
      toggle("Show all answers", false, (v) => editor.setRevealAll(v)))));
  root.append(host);
  editor = new QuestionEditor(host, { questions: quiz.questions, sourceText: quiz.source_text, request: null },
    { onChange: markDirty, canRegenerate: () => !!store.status?.ready });

  async function save(button) {
    await busy(button, async () => {
      try {
        const res = await put(`/quizzes/${id}`, { quiz });
        const { questions, ...rest } = res.quiz;
        Object.assign(quiz, rest);
        quiz.questions.splice(0, quiz.questions.length, ...questions); // the editor keeps a reference to this array
        editor.render();
        dirty = false;
        saveBtn.disabled = true;
        toast("Changes saved.", "good");
        if (res.warning) toast(res.warning, "warn", 6000);
      } catch (err) { toast(err.message, "bad", 7000); }
    });
  }
  const beforeUnload = (e) => { if (dirty) { e.preventDefault(); e.returnValue = ""; } };
  window.addEventListener("beforeunload", beforeUnload);
  return () => {
    window.removeEventListener("beforeunload", beforeUnload);
    if (dirty) toast("Unsaved changes to the quiz were discarded.", "warn");
  };
}
