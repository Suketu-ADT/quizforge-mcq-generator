// Settings is the only page that shows model tags (elsewhere the app uses the friendly names).
import { del, post, put } from "../api.js";
import { loadSettings, onStatus, refreshStatus } from "../store.js";
import { alertBox, btn, busy, clear, confirmDialog, field, h, pageHead, tag, toast } from "../ui.js";

export async function render(root) {
  root.append(pageHead("Settings", "Everything runs against the Ollama server on this computer. No API keys, no cloud."));
  const statusCard = h("section", { class: "card", style: { marginBottom: "1rem" } });
  const models = h("div", { class: "grid cols-2", style: { marginBottom: "1rem" } });
  root.append(statusCard, h("h2", { style: { margin: "1.4rem 0 .3rem" } }, "Model"),
    h("p", { class: "muted small" }, "Pick the model that writes your questions. You can switch at any time; the next generation uses it."), models);

  const unsub = onStatus((st) => { paintStatus(statusCard, st); paintModels(models, st); });
  refreshStatus(true);

  const s = await loadSettings(true);
  const url = h("input", { type: "url", value: s.base_url });
  const modelSel = h("select");
  const custom = h("input", { type: "text", placeholder: "e.g. qwen2.5:1.5b", hidden: true });
  const temp = h("input", { type: "range", min: 0, max: 1.2, step: 0.05, value: s.temperature });
  const tempOut = h("output", null, Number(s.temperature).toFixed(2));
  temp.addEventListener("input", () => { tempOut.textContent = Number(temp.value).toFixed(2); });
  const timeout = h("input", { type: "number", min: 30, max: 3600, value: s.timeout_s });
  const maxChars = h("input", { type: "number", min: 2000, max: 500000, step: 1000, value: s.max_input_chars });
  const formMsg = h("div");
  const fillModels = (st) => {
    const tags = [...new Set([s.model, ...(st?.installed_models || []), "quizforge-3b", "quizforge-7b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"])];
    clear(modelSel).append(...tags.map((t) => h("option", { value: t, selected: t === s.model }, t)), h("option", { value: "__custom" }, "Other (type a tag)..."));
  };
  const unsubSel = onStatus(fillModels);
  modelSel.addEventListener("change", () => { custom.hidden = modelSel.value !== "__custom"; if (!custom.hidden) custom.focus(); });
  const chosenModel = () => (modelSel.value === "__custom" ? custom.value.trim() : modelSel.value);

  root.append(h("h2", { style: { margin: "1.6rem 0 .6rem" } }, "Connection and generation"),
    h("section", { class: "card" }, h("div", { class: "stack" },
      h("div", { class: "grid cols-2" }, field("Ollama URL", url, "Keep it on localhost so your material never leaves this computer."),
        field("Model tag (advanced)", h("div", { class: "stack", style: { gap: ".4rem" } }, modelSel, custom))),
      h("div", { class: "grid cols-3" },
        field("Temperature", h("div", { class: "row", style: { flexWrap: "nowrap" } }, temp, tempOut), "Lower = more predictable questions."),
        field("Request timeout (seconds)", timeout),
        field("Maximum material size (characters)", maxChars)),
      formMsg,
      h("div", { class: "row" },
        btn("Save settings", { kind: "primary", icon: "check", onClick: (e) => busy(e.currentTarget, async () => {
          try {
            const res = await put("/settings", { base_url: url.value, model: chosenModel(), temperature: Number(temp.value), timeout_s: Number(timeout.value), max_input_chars: Number(maxChars.value) });
            await loadSettings(true);
            await refreshStatus(true);
            clear(formMsg).append(res.warning ? alertBox("warn", res.warning) : alertBox("good", "Settings saved."));
          } catch (err) { clear(formMsg).append(alertBox("bad", err.message)); }
        }) }),
        btn("Test connection", { icon: "refresh", onClick: (e) => busy(e.currentTarget, async () => {
          try {
            const res = await post("/settings/test", { base_url: url.value, model: chosenModel() });
            clear(formMsg).append(alertBox(res.ok ? "good" : "warn", res.message));
          } catch (err) { clear(formMsg).append(alertBox("bad", err.message)); }
        }) })))));

  root.append(h("h2", { style: { margin: "1.6rem 0 .6rem" } }, "Your data"),
    h("section", { class: "card" },
      h("p", null, "Quizzes, attempts and drafts are stored in a SQLite file on this computer:"),
      h("pre", { class: "cmd" }, s.db_path),
      h("div", { class: "row", style: { marginTop: "1rem" } },
        btn("Clear quiz history", { kind: "danger", icon: "trash", onClick: async () => {
          const ok = await confirmDialog({ title: "Clear all quiz attempts?", message: "Scores and analytics are deleted permanently. Your saved quizzes are kept.", confirm: "Delete attempts", danger: true });
          if (!ok) return;
          try { const r = await del("/attempts"); toast(`Deleted ${r.deleted} attempt(s).`, "good"); } catch (err) { toast(err.message, "bad"); }
        } }))));
  return () => { unsub(); unsubSel(); };
}

function paintStatus(card, st) {
  clear(card);
  const state = st.server_down ? ["bad", "App server stopped"] : !st.connected ? ["bad", "Ollama is not running"] : !st.ready ? ["warn", "Model not installed"] : ["good", "Connected and ready"];
  card.append(h("div", { class: "row between" },
    h("div", { class: "row" }, h("h2", null, "Runtime status"), tag(state[1], state[0])),
    btn("Refresh", { size: "sm", icon: "refresh", onClick: (e) => busy(e.currentTarget, () => refreshStatus(true)) })));
  const rows = [["Ollama URL", st.base_url], ["Active model", st.model ? `${st.model_name} (${st.model})` : "-"]];
  if (st.version) rows.push(["Ollama version", st.version]);
  if (st.connected) rows.push(["Installed models", st.installed_models.join(", ") || "none"]);
  card.append(h("table", { class: "table", style: { marginTop: ".6rem" } }, h("tbody", null, rows.map(([k, v]) => h("tr", null, h("th", { style: { width: "190px" } }, k), h("td", { class: "mono" }, v))))));
  if (!st.connected && !st.server_down) {
    card.append(h("div", { style: { marginTop: ".8rem" } }, alertBox("warn", st.error || "Cannot reach Ollama.",
      h("div", { class: "small", style: { marginTop: ".3rem" } }, "Open the Ollama app, or run this in a terminal:"), h("pre", { class: "cmd" }, "ollama serve"))));
  }
  if (st.connected && st.local === false) card.append(h("div", { style: { marginTop: ".8rem" } }, alertBox("warn", "This Ollama server is not on this computer: your study material is sent there.")));
}

function paintModels(grid, st) {
  clear(grid);
  for (const m of st.models || []) {
    let pill = null;
    if (m.active) pill = tag("In use", "primary");
    else if (m.installed) pill = tag("Installed", "good");
    else if (m.installed === false) pill = tag("Not installed", "medium");
    const how = m.installed === false ? (m.finetuned
      ? h("div", { class: "small" }, m.gguf_present
        ? ["The file is in model_files. Import it once with ", h("b", null, "setup_models.bat"), " or:"]
        : ["Put ", h("code", null, m.gguf), " in the ", h("code", null, "model_files"), " folder, then run ", h("b", null, "setup_models.bat"), "."],
        m.gguf_present ? h("pre", { class: "cmd" }, `python -m finetune.import_to_ollama --gguf model_files\\${m.gguf} --name ${m.tag} --base ${m.base}`) : null)
      : h("div", { class: "small" }, "Install it once (needs internet):", h("pre", { class: "cmd" }, `ollama pull ${m.tag}`))) : null;
    grid.append(h("article", { class: `card model-card ${m.active ? "active" : ""}` },
      h("div", { class: "name" }, m.tier, m.finetuned ? tag("Fine-tuned", "primary") : tag("Base")),
      h("div", { class: "row" }, h("code", null, m.tag), h("span", { class: "small muted" }, `download ${m.size}`), pill),
      h("div", { class: "desc" }, m.blurb), how,
      h("div", null, btn(m.active ? "In use" : "Use this model", { kind: m.active ? "" : "primary", size: "sm", disabled: m.active || m.installed === false,
        onClick: (e) => busy(e.currentTarget, async () => {
          try { await post("/model", { tag: m.tag }); await loadSettings(true); await refreshStatus(true); toast(`Switched to the ${m.tier} model.`, "good"); }
          catch (err) { toast(err.message, "bad"); }
        }) }))));
  }
}
