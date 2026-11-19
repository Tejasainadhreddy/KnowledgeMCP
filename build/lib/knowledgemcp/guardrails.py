"""Guardrails: check what comes in, clean what we retrieve, verify what goes out.

1. Input validation: length, control characters, obvious prompt-injection phrases.
2. Retrieved text: documents are untrusted. Lines that look like instructions to the model are removed,
   and every source is wrapped in <source> tags that the system prompt tells the model to treat as data.
3. Output: every [S#] citation must point to a source we actually sent; answers with no valid citation
   are refused instead of returned.

Pattern matching will not stop every attack. It is one layer; the others are the system prompt,
team filtering in SQL (the model never sees other teams' data) and citation checking.
"""

from __future__ import annotations

import html
import re
import unicodedata

from .models import Hit

MAX_QUESTION_CHARS = 1000
MIN_QUESTION_CHARS = 3

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules|messages)",
    r"disregard\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier|system)",
    r"forget\s+(all\s+|everything\s+|your\s+)(previous\s+)?(instructions|rules)",
    r"\byou\s+are\s+now\b",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak",
    r"(reveal|show|print|repeat|leak)\s+(me\s+)?(your|the)\s+(system\s+prompt|instructions|hidden\s+prompt)",
    r"\bsystem\s+prompt\b",
    r"<\s*/?\s*(system|assistant|instructions?|source)\s*>",
    r"\bact\s+as\s+(an?\s+)?(unrestricted|unfiltered|evil)\b",
]
_INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)
_CITATION = re.compile(r"\[S(\d+)\]")
REMOVED = "[removed by guardrail: possible prompt injection]"


class GuardrailViolation(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


def validate_question(question: str) -> str:
    if not isinstance(question, str):
        raise GuardrailViolation("invalid_input", "Question must be text.")
    cleaned = "".join(ch for ch in question if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < MIN_QUESTION_CHARS:
        raise GuardrailViolation("too_short", f"Question must be at least {MIN_QUESTION_CHARS} characters.")
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise GuardrailViolation("too_long", f"Question must be at most {MAX_QUESTION_CHARS} characters.")
    if looks_like_injection(cleaned):
        raise GuardrailViolation("prompt_injection", "The question looks like an attempt to override instructions.")
    return cleaned


def sanitize_retrieved(text: str) -> tuple[str, bool]:
    """Remove sentences that try to instruct the model. Returns (clean_text, was_flagged)."""
    flagged = False
    out_lines = []
    for line in text.splitlines():
        if looks_like_injection(line):
            flagged = True
            out_lines.append(REMOVED)
        else:
            out_lines.append(line)
    return "\n".join(out_lines), flagged


def format_sources(hits: list[Hit]) -> str:
    """Wrap each chunk in <source> tags. Text is HTML-escaped so a document cannot close the tag early."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        attrs = f'id="S{i}" title="{html.escape(h.title)}" location="{html.escape(h.location())}"'
        blocks.append(f"<source {attrs}>\n{html.escape(h.text, quote=False)}\n</source>")
    return "\n\n".join(blocks)


def check_citations(answer: str, num_sources: int) -> tuple[str, list[int]]:
    """Drop citations that point at sources we never sent. Returns (clean_answer, valid_source_numbers)."""
    valid: list[int] = []

    def keep(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= num_sources:
            if n not in valid:
                valid.append(n)
            return m.group(0)
        return ""

    cleaned = _CITATION.sub(keep, answer)
    return re.sub(r"\s+([.,;:])", r"\1", cleaned).strip(), valid
