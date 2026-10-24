# Viva preparation - QuizForge (PS 97)

Short, accurate answers to the questions a jury is likely to ask. Never claim the answer key is guaranteed correct.

## Concepts

**Why Qwen 2.5?**
It is an open-weight model family (0.5B to 72B parameters) with good instruction following and JSON output. Several sizes (3B, 7B, 14B) run on student hardware through Ollama, and it can run fully locally.

**What is Ollama?**
A local runtime that downloads, manages and serves open LLMs through a simple REST API (`/api/chat`, `/api/tags`) on `localhost:11434`.

**Why a local LLM?**
Privacy (study material never leaves the machine during inference), no API keys or per-token cost, a reproducible setup, and no cloud account. Trade-offs: it needs hardware, and it is slower and weaker than the largest cloud models.

**Is calling Ollama an "API call"?**
The app makes no call to any cloud service. It talks to a program running on the same computer over `localhost`. If your course rules also exclude local servers, confirm with your faculty.

## Pipeline

**How are MCQs generated?**
The text is cleaned and chunked. For each batch of up to 3 questions the app sends a system prompt (12 rules), the chunk inside `<material>` tags, the user's settings and a JSON schema (Ollama structured output). The model returns questions with options, answer, explanation, difficulty, topic and a supporting excerpt. Each accepted question is shown immediately.

**How are long documents processed?**
Text is split at paragraph and heading boundaries into chunks of about 3,500 characters. The requested count is spread over evenly spaced chunks, earlier questions are passed as "do not repeat", and every question records its source section. Input above the configured limit is refused, never silently truncated.

**How are answers validated?**
Safe JSON parsing, Pydantic schema checks (option count, unique ids, valid correct answer, difficulty), a check that the quoted source sentence exists in the text, duplicate detection, then human review. This proves the output is well-formed and grounded. It does not prove the key is correct.

**What happens when the model produces invalid JSON?**
The parser fails safely and the reason is fed back in a repair prompt. Each batch gets at most 2 repairs, then a bounded top-up on unused chunks. Invalid items are discarded, the shortfall is reported, and the user can generate the remainder. There are no infinite loops.

**What does the answer-key double-check do?**
The model answers each question again without seeing the key. A different answer adds a "review the key" badge. The same model wrote and checked the key, so agreement is not proof and disagreement is not proof of error; it only shows where to look first.

**How is prompt injection handled?**
Uploaded text is untrusted. It is delimited as data, the system prompt says to ignore instructions inside it, output must match a schema, and nothing the model returns is ever executed.

## Quiz and metrics

**How is quiz accuracy calculated?**
`percent = correct / total x 100`. Unanswered questions count as not correct. It is computed from the stored answer key, never from a new LLM call. Topic and difficulty accuracy use the same rule per group.

**How do you measure the LLM's performance?**
Speed: tokens per second (`eval_count / (eval_duration / 1e9)`) and seconds per accepted question. Reliability: valid-JSON rate and acceptance rate (`accepted / (accepted + rejected)`). Grounding rate. Key accuracy needs a human to judge each key. Self-consistency comes from the double-check. Compare models on the same fixed test set and report the hardware.

**Quiz accuracy versus model accuracy?**
Quiz accuracy measures the student. Model quality is measured by the metrics above.

## Honest limitations
Possible wrong keys; weaker 3B models can hallucinate or write ambiguous distractors; scanned PDFs need OCR; speed depends on hardware; quality depends on the source text; single-user local app.

## Suggested demo (about 5 minutes)
1. Open the app and show the **Connected** status.
2. Generate Quiz: upload a short PDF and choose 5 questions, Medium.
3. Watch questions stream in; show Stop.
4. Review: reveal an answer, edit one, run Double-check.
5. Save and take the quiz; show the instant feedback.
6. Show results, charts and Retry missed.
7. Export the question paper and the answer key as PDF.
