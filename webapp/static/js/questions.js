// Question cards with review tools: reveal, edit (re-validated by the server), regenerate with the model, delete,
// and add a question by hand. Used by the Generate page (draft) and the Question Bank editor.
import { post } from "./api.js";
import { alertBox, btn, busy, clear, confirmDialog, difficultyTag, field, h, icon, tag, toast } from "./ui.js";

const LETTERS = "ABCDEFGH";

export function keyCheckTag(q) {
  if (q.key_check === "agrees") return tag("Key confirmed", "good");
  if (q.key_check === "disagrees") return tag("Review the key", "bad");
  return null;
}

// Read-only card (used while questions stream in, and in quiz reviews)
export function questionCard(q, n, { reveal = false, showSource = true, enter = false } = {}) {
  const card = h("article", { class: `q-card ${enter ? "enter" : ""} ${q.key_check === "disagrees" ? "flag-bad" : ""}` });
  card.append(questionBody(q, n, { reveal, showSource }));
  return card;
}

function questionBody(q, n, { reveal, showSource }) {
  const frag = document.createDocumentFragment();
  frag.append(h("div", { class: "q-top" }, h("span", { class: "q-num" }, `Q${n}`), difficultyTag(q.difficulty),
    q.topic && q.topic !== "General" ? tag(q.topic) : null, keyCheckTag(q)));
  frag.append(h("div", { class: "q-text" }, q.question));
  frag.append(h("ul", { class: "q-opts" }, q.options.map((o) =>
    h("li", { class: reveal && o.id === q.correct_answer ? "correct" : "" }, h("span", { class: "letter" }, o.id),
      h("span", null, o.text), reveal && o.id === q.correct_answer ? h("span", { class: "spacer" }) : null,
      reveal && o.id === q.correct_answer ? icon("check") : null))));
  if (reveal && (q.explanation || (showSource && q.source_excerpt))) {
    frag.append(h("div", { class: "q-extra" },
      q.explanation ? h("div", null, h("span", { class: "label" }, "Explanation"), q.explanation) : null,
      showSource && q.source_excerpt ? h("div", { class: "q-source" }, h("span", { class: "label" }, q.source_section ? `Source · ${q.source_section}` : "Source"), `“${q.source_excerpt}”`) : null));
  }
  return frag;
}

/**
 * Editable list. `state` = { questions, sourceText, request }; `onChange(questions)` is called after every change.
 */
export class QuestionEditor {
  constructor(host, state, { onChange, showSource = true, canRegenerate = () => true, numOptions = 4 } = {}) {
    this.host = host;
    this.state = state;
    this.onChange = onChange || (() => {});
    this.showSource = showSource;
    this.canRegenerate = canRegenerate;
    this.numOptions = numOptions;
    this.revealed = new Set();
    this.revealAll = false;
    this.editing = new Set();
    this.list = h("div", { class: "q-list" });
    this.addSlot = h("div");
    host.append(this.list, this.addSlot);
    this.render();
  }

  get questions() { return this.state.questions; }

  setRevealAll(v) { this.revealAll = v; this.render(); }

  changed() { this.onChange(this.questions); }

  render() {
    clear(this.list);
    this.questions.forEach((q, i) => this.list.append(this.card(q, i)));
    clear(this.addSlot).append(h("div", { class: "row", style: { marginTop: ".9rem" } },
      btn("Add a question", { icon: "plus", kind: "ghost", onClick: () => this.openAdd() })));
  }

  card(q, i) {
    const card = h("article", { class: `q-card ${q.key_check === "disagrees" ? "flag-bad" : ""}`, dataset: { id: q.id } });
    if (this.editing.has(q.id)) {
      card.append(this.form(q, (saved) => { this.questions[i] = saved; this.editing.delete(q.id); this.render(); this.changed(); },
        () => { this.editing.delete(q.id); this.render(); }));
      return card;
    }
    const reveal = this.revealAll || this.revealed.has(q.id);
    card.append(questionBody(q, i + 1, { reveal, showSource: this.showSource }));
    const actions = h("div", { class: "q-actions" },
      btn(reveal ? "Hide answer" : "Show answer", { size: "sm", kind: "ghost", icon: reveal ? "eyeOff" : "eye", onClick: () => {
        if (this.revealed.has(q.id)) this.revealed.delete(q.id); else this.revealed.add(q.id);
        if (this.revealAll && reveal) { this.revealAll = false; this.questions.forEach((x) => x.id !== q.id && this.revealed.add(x.id)); }
        this.render();
      } }),
      btn("Edit", { size: "sm", kind: "ghost", icon: "edit", onClick: () => { this.editing.add(q.id); this.render(); } }),
      this.canRegenerate() ? btn("Regenerate", { size: "sm", kind: "ghost", icon: "refresh", onClick: (e) => this.regenerate(i, card, e.currentTarget) }) : null,
      h("span", { class: "spacer" }),
      btn("Delete", { size: "sm", kind: "ghost danger", icon: "trash", onClick: () => this.remove(i) }));
    card.append(actions);
    return card;
  }

  async regenerate(i, card, button) {
    const q = this.questions[i];
    card.classList.add("busy");
    await busy(button, async () => {
      try {
        const res = await post("/questions/regenerate", {
          question: q, others: this.questions.filter((x) => x.id !== q.id),
          source_text: this.state.sourceText || "", request: this.state.request || null,
        });
        this.questions[i] = res.question;
        this.revealed.add(res.question.id);
        toast(`Question ${i + 1} was rewritten.`, "good");
        this.render();
        this.changed();
      } catch (err) {
        card.classList.remove("busy");
        toast(err.message, "bad", 6000);
      }
    });
  }

  async remove(i) {
    const ok = await confirmDialog({ title: `Delete question ${i + 1}?`, message: "It is removed from this quiz. This cannot be undone once the quiz is saved.", confirm: "Delete", danger: true });
    if (!ok) return;
    this.questions.splice(i, 1);
    this.render();
    this.changed();
  }

  openAdd() {
    const n = this.questions[0]?.options?.length || this.numOptions || 4;
    const blank = { id: "", question: "", options: Array.from({ length: n }, (_, k) => ({ id: LETTERS[k], text: "" })),
      correct_answer: "A", explanation: "", difficulty: "Medium", topic: this.questions[0]?.topic || "General", source_excerpt: "" };
    const card = h("article", { class: "q-card enter" });
    card.append(h("h3", { style: { marginBottom: ".7rem" } }, "New question"),
      this.form(blank, (saved) => { this.questions.push(saved); this.render(); this.changed(); toast("Question added.", "good"); },
        () => this.render()));
    clear(this.addSlot).append(card);
    card.querySelector("textarea")?.focus();
  }

  form(q, onSaved, onCancel) {
    const draft = structuredClone(q);
    const wrap = h("div", { class: "q-edit" });
    const err = h("div");
    const qText = h("textarea", { rows: 3, value: draft.question, placeholder: "Write the question...", "aria-label": "Question text" });
    const optsBox = h("div", { class: "stack", style: { gap: ".45rem" } });
    const radioName = `correct-${Math.random().toString(36).slice(2)}`;
    const renderOpts = () => {
      clear(optsBox);
      draft.options.forEach((o, k) => {
        o.id = LETTERS[k];
        const input = h("input", { type: "text", value: o.text, placeholder: `Option ${o.id}`, "aria-label": `Option ${o.id}`, oninput: (e) => { o.text = e.target.value; } });
        optsBox.append(h("div", { class: "opt-edit" },
          h("input", { type: "radio", name: radioName, checked: draft.correct_answer === o.id, title: "Mark as the correct answer", "aria-label": `Option ${o.id} is correct`, onchange: () => { draft.correct_answer = o.id; } }),
          h("span", { class: "letter" }, o.id), input,
          draft.options.length > 2 ? h("button", { type: "button", class: "icon-btn", title: "Remove option", "aria-label": `Remove option ${o.id}`, onclick: () => {
            draft.options.splice(k, 1);
            if (!draft.options.some((x, j) => LETTERS[j] === draft.correct_answer)) draft.correct_answer = "A";
            renderOpts();
          } }, icon("x")) : null));
      });
      if (draft.options.length < 6) optsBox.append(h("div", null, btn("Add option", { size: "sm", kind: "ghost", icon: "plus", onClick: () => { draft.options.push({ id: LETTERS[draft.options.length], text: "" }); renderOpts(); } })));
    };
    renderOpts();
    const diff = h("select", { "aria-label": "Difficulty" }, ["Easy", "Medium", "Hard"].map((d) => h("option", { value: d, selected: draft.difficulty === d }, d)));
    const topic = h("input", { type: "text", value: draft.topic || "" });
    const expl = h("textarea", { rows: 2, value: draft.explanation || "", placeholder: "Why is this the right answer? (optional)" });
    wrap.append(
      field("Question", qText),
      h("div", null, h("div", { class: "small muted", style: { marginBottom: ".4rem" } }, "Options · select the radio button of the correct answer"), optsBox),
      h("div", { class: "grid cols-2" }, field("Difficulty", diff), field("Topic", topic)),
      field("Explanation", expl),
      err,
      h("div", { class: "row end" },
        btn("Cancel", { kind: "ghost", onClick: onCancel }),
        btn("Save question", { kind: "primary", icon: "check", onClick: async (e) => {
          const payload = { ...draft, question: qText.value, difficulty: diff.value, topic: topic.value, explanation: expl.value,
            options: draft.options.map((o) => ({ id: o.id, text: o.text })) };
          if (!payload.id) delete payload.id;
          await busy(e.currentTarget, async () => {
            try {
              const res = await post("/questions/validate", { question: payload, others: this.questions });
              clear(err);
              if (res.warning) toast(res.warning, "warn");
              onSaved(res.question);
            } catch (ex) {
              clear(err).append(alertBox("bad", ex.message));
            }
          });
        } })),
    );
    return wrap;
  }
}
