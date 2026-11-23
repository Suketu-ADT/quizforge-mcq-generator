import { go } from "../nav.js";
import { btn, h, icon, pageHead } from "../ui.js";

export function render(root) {
  root.append(pageHead("About QuizForge", "Problem statement PS 97 — Automatic MCQ Quiz Generator using a local Qwen 2.5 model."));
  const point = (ic, title, text) => h("div", { class: "card" }, h("div", { class: "row", style: { marginBottom: ".4rem" } },
    h("span", { class: "step" }, icon(ic)), h("h3", null, title)), h("p", { class: "muted small", style: { margin: 0 } }, text));
  root.append(
    h("section", { class: "card pad-lg", style: { marginBottom: "1rem" } },
      h("h2", { style: { marginBottom: ".6rem" } }, "How a question is made"),
      h("div", { class: "pipeline" },
        h("span", null, "Your notes"), h("i", null, "→"), h("span", null, "Extract and clean"), h("i", null, "→"),
        h("span", null, "Remove AI instructions"), h("i", null, "→"), h("span", null, "Qwen 2.5 writes JSON"), h("i", null, "→"),
        h("span", { class: "key" }, "Validator checks"), h("i", null, "→"), h("span", null, "You review"), h("i", null, "→"), h("span", null, "Quiz and analytics")),
      h("p", { class: "muted", style: { marginTop: "1rem" } },
        "The model only proposes questions. Ordinary code decides what is kept: the reply must be valid JSON in a fixed schema, every question must quote the sentence of your material it comes from, duplicates are rejected, and a rejected batch is retried at most twice. Scores always come from the saved answer key, never from a new model call.")),
    h("div", { class: "grid cols-3", style: { marginBottom: "1rem" } },
      point("shield", "Private by design", "The model runs in Ollama on this computer. No OpenAI, Gemini or other cloud API, and no API key. The app never downloads models by itself."),
      point("target", "Grounded questions", "Each question carries a word-for-word quote from your material, so you can see where it came from. Questions that cannot be traced back are rejected."),
      point("bolt", "Fine-tuned models", "Qwen 2.5 3B and 7B were fine-tuned with QLoRA on SciQ plus checked teacher questions, in this app's exact prompt and JSON format."),
      point("alert", "Safe against injected instructions", "Sentences in an upload that try to give orders to the AI (“ignore previous instructions...”) are removed before the model sees the text."),
      point("eye", "Human in the loop", "Answer keys are written by the model, so you review, edit or regenerate questions before a quiz is used. An optional blind re-answer flags doubtful keys."),
      point("chart", "Measured honestly", "The evaluation suite reports 12 metrics with worked calculations and compares base and fine-tuned models with the same judge.")),
    h("section", { class: "card" },
      h("h2", { style: { marginBottom: ".5rem" } }, "Technology"),
      h("p", { class: "muted" }, "Front end: HTML, CSS and JavaScript (no framework, works offline). Back end: Python with Flask. Model runtime: Ollama with Qwen 2.5. Storage: SQLite. Validation: Pydantic. Documents: PyMuPDF and python-docx."),
      h("h2", { style: { margin: "1rem 0 .5rem" } }, "Limitations"),
      h("ul", { class: "muted", style: { margin: 0, paddingLeft: "1.1rem" } },
        h("li", null, "A small model can still mark the wrong answer: always review the keys."),
        h("li", null, "Scanned PDFs (pictures of text) need OCR, which is not included."),
        h("li", null, "Speed depends on your computer; the 7B models need about 8 GB of free memory.")),
      h("div", { class: "row", style: { marginTop: "1.2rem" } }, btn("Generate a quiz", { kind: "primary", icon: "sparkles", onClick: () => go("generate") }))));
}
