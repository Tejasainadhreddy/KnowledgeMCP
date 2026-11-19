"""Split sections into overlapping chunks without losing their page/section labels.

We split on paragraph boundaries first, then sentences, so a chunk rarely cuts a sentence in half.
Chunks never cross a section or page boundary, which keeps citations exact.
"""

from __future__ import annotations

import re

from .models import Chunk, Section

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _pieces(text: str, max_chars: int) -> list[str]:
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            out.append(para)
            continue
        for sentence in _SENTENCE_END.split(para):
            while len(sentence) > max_chars:  # very long sentence: hard split
                out.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if sentence.strip():
                out.append(sentence.strip())
    return out


def split_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    if max_chars <= overlap:
        raise ValueError("max_chars must be larger than overlap")
    chunks: list[str] = []
    current = ""
    for piece in _pieces(text, max_chars):
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # start the overlap at a word boundary
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            current = f"{tail} {piece}".strip() if tail else piece
            if len(current) > max_chars:
                current = piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def chunk_sections(
    doc_id: str, team: str, sections: list[Section], max_chars: int = 1200, overlap: int = 150
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        for text in split_text(section.text, max_chars, overlap):
            header = f"[{section.section}]\n" if section.section else ""
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_index=len(chunks),
                    text=header + text,
                    team=team,
                    page=section.page,
                    section=section.section,
                )
            )
    return chunks
