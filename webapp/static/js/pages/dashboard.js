import { get } from "../api.js";
import { go } from "../nav.js";
import { onStatus } from "../store.js";
import { alertBox, btn, fmtDate, fmtWhen, h, icon, plural, tag } from "../ui.js";

export async function render(root) {
  const statusSlot = h("div");
  root.append(
    h("section", { class: "hero" },
      h("h1", null, "Turn your notes into a quiz"),
      h("p", null, "Upload a PDF, Word file or text. Your local AI model writes multiple-choice questions, every one is checked against your material, and you review them before anyone takes the quiz."),
      h("div", { class: "row" },
        btn("Generate a quiz", { kind: "light lg", icon: "sparkles", onClick: () => go("generate") }),
        btn("Take a quiz", { kind: "ghost-light lg", icon: "play", onClick: () => go("quiz") }))),
    statusSlot,
  );
  const body = h("div", { class: "stack", style: { marginTop: "1.2rem" } }, h("div", { class: "skeleton", style: { height: "110px" } }));
  root.append(body);

  const unsub = onStatus((st) => {
    statusSlot.replaceChildren();
    if (st.server_down) return;
    if (!st.connected) {
      statusSlot.append(h("div", { class: "card", style: { marginTop: "1.2rem" } },
        h("h2", null, "Finish setting up"),
        h("ol", { class: "steps-list" },
          h("li", null, h("div", null, h("b", null, "Install Ollama"), h("div", { class: "muted small" }, "Download it from ollama.com and open it once."))),
          h("li", null, h("div", null, h("b", null, "Install a model"), h("div", { class: "muted small" }, "Run setup_models.bat in the project folder, or see Settings for the commands."))),
          h("li", null, h("div", null, h("b", null, "Come back here"), h("div", { class: "muted small" }, "This page notices when Ollama is running.")))),
        h("div", { class: "row", style: { marginTop: "1rem" } }, btn("Open Settings", { icon: "settings", onClick: () => go("settings") }))));
    } else if (!st.ready) {
      statusSlot.append(h("div", { style: { marginTop: "1.2rem" } }, alertBox("warn", "The selected model is not installed yet. ",
        h("a", { href: "#/settings" }, "Choose an installed model or see how to install it."))));
    }
  });

  let data;
  try {
    [data] = await Promise.all([get("/dashboard")]);
  } catch (err) {
    body.replaceChildren(alertBox("bad", err.message));
    return unsub;
  }
  let session = null;
  try { session = (await get("/session")).session; } catch { /* optional */ }

  body.replaceChildren();
  const cont = [];
  if (data.draft) {
    cont.push(h("div", { class: "card row between" },
      h("div", null, h("div", { class: "tiny muted" }, "UNSAVED DRAFT"), h("h3", null, data.draft.title), h("div", { class: "small muted" }, `${plural(data.draft.questions, "question")} waiting for review`)),
      btn("Continue reviewing", { kind: "primary", icon: "edit", onClick: () => go("generate") })));
  }
  if (session) {
    cont.push(h("div", { class: "card row between" },
      h("div", null, h("div", { class: "tiny muted" }, "UNFINISHED QUIZ"), h("h3", null, session.quiz_title),
        h("div", { class: "small muted" }, `${Object.keys(session.answers || {}).length} of ${session.total} answered`)),
      btn("Resume", { kind: "primary", icon: "play", onClick: () => go("quiz/resume") })));
  }
  if (cont.length) body.append(h("div", { class: "grid cols-2" }, ...cont));

  const s = data.stats;
  const pct = (v) => (v == null ? "-" : `${Math.round(v)}%`);
  body.append(h("div", { class: "grid cols-4" },
    statCard("Quizzes saved", s.quizzes, "in your question bank"),
    statCard("Questions", s.questions_in_bank, `${s.questions_generated.toLocaleString()} written by the model so far`),
    statCard("Quiz attempts", s.attempts, "taken on this computer"),
    statCard("Average score", pct(s.avg_percent), s.best_percent == null ? "no attempts yet" : `best ${pct(s.best_percent)}`)));

  const recent = h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", null, "Recent quizzes"), h("span", { class: "spacer" }),
    btn("Open bank", { size: "sm", kind: "ghost", onClick: () => go("bank") })));
  if (!data.recent_quizzes.length) {
    recent.append(h("p", { class: "muted" }, "No quizzes yet. Generate your first one from your notes."));
  } else {
    recent.append(h("div", { class: "stack", style: { gap: ".2rem" } }, data.recent_quizzes.map((q) =>
      h("div", { class: "row between", style: { padding: ".5rem 0", borderBottom: "1px solid var(--border)" } },
        h("div", null, h("b", null, q.title), h("div", { class: "small muted" }, `${plural(q.n_questions, "question")} · ${fmtDate(q.updated_at)}`)),
        h("div", { class: "row" }, q.best_percent != null ? tag(`best ${Math.round(q.best_percent)}%`, "primary") : null,
          btn("Take", { size: "sm", icon: "play", onClick: () => go(`quiz/${q.id}`) }))))));
  }
  const activity = h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", null, "Activity")));
  if (!data.activity.length) activity.append(h("p", { class: "muted" }, "Nothing yet."));
  else activity.append(h("div", { class: "stack", style: { gap: ".55rem" } }, data.activity.map((a) =>
    h("div", { class: "row", style: { alignItems: "flex-start", flexWrap: "nowrap" } }, h("span", { class: "muted", style: { marginTop: ".1rem" } }, icon(ACT_ICON[a.kind] || "info")),
      h("div", null, h("div", { class: "small" }, a.message), h("div", { class: "tiny muted" }, fmtWhen(a.ts)))))));
  body.append(h("div", { class: "grid cols-2" }, recent, activity));
  return unsub;
}

const ACT_ICON = { generate: "sparkles", save: "library", attempt: "target", delete: "trash", settings: "settings", history: "trash" };

function statCard(label, value, hint) {
  return h("div", { class: "card stat" }, h("span", { class: "label" }, label), h("span", { class: "value" }, typeof value === "number" ? value.toLocaleString() : value), h("span", { class: "hint" }, hint));
}
