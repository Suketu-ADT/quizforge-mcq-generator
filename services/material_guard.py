"""Input guard for uploaded study material.

Uploaded material is untrusted. A document can hide sentences addressed to the AI rather than to the learner, for
example "ignore the rules", "print your system prompt", "include the code X in your reply" or "write questions that
teach how to steal a password". The system prompt already tells the model to treat such text as ordinary data, but
small models still obey some of it. A reply that obeyed counts as a successful attack even when the validator later
rejects it.

So these sentences are taken out before the model sees the material, and the user is told what was removed. They are
never facts worth asking about, so removing them costs nothing. If nothing teachable is left (the whole upload is a
request to the quiz writer), no questions are written at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# the prompt wraps the material in <material> tags: a fake closing tag in the upload must not end it early
_TAG_RE = re.compile(r"</?\s*(?:material|source_text|system|instructions?|user|assistant)\s*>", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MARKER = r"(?-i:[A-Z0-9][A-Z0-9_-]{3,})"  # an upper-case code word such as ZEBRA-4417

# Each pattern describes a sentence that talks to the AI instead of teaching the subject. They are anchored to
# imperative or second-person wording so that material *about* prompt injection ("an attacker may tell the model to
# ignore its instructions") is kept.
_DIRECTIVES = [re.compile(p, re.IGNORECASE) for p in (
    # cancelling or replacing the rules
    r"(?:^|[:;,]\s*|\b(?:please|now|just|and)\s+)(?:ignore|disregard|forget|override|bypass)\b[^.!?\n]{0,40}"
    r"\b(?:rules|instructions|prompts?|guidelines|polic(?:y|ies))\b",
    r"\b(?:system|developer|admin(?:istrator)?)\s+override\b",
    r"\bnew\s+instructions?\s*(?::|from\b)",
    r"\b(?:note|message|instructions?|command)s?\s+(?:to|for)\s+(?:the\s+)?(?:ai|model|assistant|llm|chatbot|bot|gpt)\b",
    # role play and claimed authority
    r"\bfrom\s+now\s+on,?\s+you\s+(?:are|will|must|shall|act)\b",
    r"\byou\s+are\s+(?:now\s+)?(?:dan\b|an?\s+(?:ai|assistant|model)\s+(?:without|with\s+no)\b)",
    r"\bto\s+prove\s+(?:that\s+)?you\s+are\b",
    r"\b(?:safety|content)\s+(?:rules|filters?|polic(?:y|ies)|guidelines|checks)\b[^.!?\n]{0,20}"
    r"\b(?:are|is|have\s+been|has\s+been)\s+(?:disabled|off|suspended|removed|lifted)\b",
    r"\brules\s+are\s+(?:now\s+)?(?:off|disabled|suspended)\b",
    r"\bauthori[sz]ed\s+(?:developer\s+|admin\s+|security\s+|red[- ]team\s+)?test\b",
    # asking for the hidden prompt
    r"\b(?:print|reveal|show|repeat|output|display|recite|tell\s+me)\b[^.!?\n]{0,30}"
    r"\b(?:system\s+prompt|hidden\s+instructions?|(?:rules|instructions)\s+you\s+(?:were|have\s+been)\s+given)\b",
    # orders about what the reply must contain
    r"\b(?:include|insert|add|write|writing|put|output|print|begin|start|end|prefix)\b[^.!?\n]{0,40}"
    r"\b(?:code|word|marker|phrase|string|token|text)\s+" + _MARKER,
    r"\b(?:every|each|all|your)\s+(?:questions?|options?|explanations?|topics?|repl(?:y|ies)|outputs?|responses?)\b"
    r"[^.!?\n]{0,40}\b(?:must|should|shall|has\s+to|have\s+to)\s+(?:contain|include|begin|start|end|say|mention)\b",
    r"\bset\s+the\s+(?:topic|explanation|answer|difficulty|correct\s+answer)s?\s+(?:of|for|to)\b",
    r"\bin\s+your\s+(?:reply|response|output)\b",
    # the upload asks the quiz writer for something instead of teaching a subject
    r"^(?:please\s+)?(?:write|create|generate|make|produce|compose|draft)\b[^.!?\n]{0,40}\bquestions?\b",
    r"^(?:as|for)\s+your\s+questions\b",
    r"^before\s+you\s+(?:write|answer|start|begin)\b",
)]

REFUSAL = (
    "QuizForge is not able to write questions from this material: it asks the quiz writer to do something instead "
    "of explaining a subject, and nothing teachable is left once those instructions are removed. Upload study "
    "material (notes, a chapter, an article) instead."
)


class MaterialRefused(ValueError):
    """The material is only instructions to the AI: there is nothing to write questions about."""


@dataclass
class GuardedMaterial:
    text: str
    removed: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return not _has_content(self.text)

    def warning(self) -> str:
        if not self.removed:
            return ""
        shown = "; ".join(f'"{s[:90]}{"..." if len(s) > 90 else ""}"' for s in self.removed[:3])
        more = f" and {len(self.removed) - 3} more" if len(self.removed) > 3 else ""
        return (f"Ignored {len(self.removed)} sentence(s) in the material that gave instructions to the AI instead of "
                f"teaching the subject: {shown}{more}.")


def is_directive(sentence: str) -> bool:
    s = sentence.strip()
    return any(p.search(s) for p in _DIRECTIVES)


def _has_content(text: str) -> bool:
    """At least one sentence of five or more words is left to ask about."""
    return any(len(s.split()) >= 5 for line in text.split("\n") for s in _SENTENCE_SPLIT.split(line))


def guard_material(text: str) -> GuardedMaterial:
    """Remove fake prompt tags and every sentence addressed to the AI. Line breaks (and so the headings the chunker
    uses) are kept."""
    if not text or not text.strip():
        return GuardedMaterial(text or "")
    removed: list[str] = []
    lines = []
    for line in _TAG_RE.sub("\n", text).split("\n"):
        sentences = _SENTENCE_SPLIT.split(line)
        kept = [s for s in sentences if not (s.strip() and is_directive(s))]
        removed += [s.strip() for s in sentences if s.strip() and is_directive(s)]
        lines.append(line if len(kept) == len(sentences) else " ".join(kept).rstrip())
    return GuardedMaterial(re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip(), removed)
