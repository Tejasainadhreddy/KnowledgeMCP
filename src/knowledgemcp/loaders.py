"""Turn raw files into Sections that remember where they came from (PDF page or Markdown heading)."""

from __future__ import annotations

import io
import re

from pypdf import PdfReader

from .models import Section

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


class UnsupportedFileType(ValueError):
    pass


def extension(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot != -1 else ""


def load_pdf(data: bytes) -> list[Section]:
    reader = PdfReader(io.BytesIO(data))
    sections = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(Section(text=text, page=number))
    return sections


def load_markdown(text: str) -> list[Section]:
    """Split on headings. The section name is the heading path, e.g. 'On-call > Paging'."""
    sections: list[Section] = []
    path: list[str] = []
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(Section(text=body, section=" > ".join(path) or None))
        buffer.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path[:] = path[: level - 1] + [match.group(2).strip()]
        else:
            buffer.append(line)
    flush()
    return sections


def load_text(text: str) -> list[Section]:
    text = text.strip()
    return [Section(text=text)] if text else []


def load_document(filename: str, data: bytes) -> list[Section]:
    ext = extension(filename)
    if ext == ".pdf":
        return load_pdf(data)
    if ext in {".md", ".markdown"}:
        return load_markdown(data.decode("utf-8", errors="replace"))
    if ext == ".txt":
        return load_text(data.decode("utf-8", errors="replace"))
    raise UnsupportedFileType(f"Unsupported file type: {filename}")


def title_from(filename: str, sections: list[Section]) -> str:
    """Use the first Markdown H1 if there is one, otherwise a cleaned-up file name."""
    for s in sections:
        if s.section:
            return s.section.split(" > ")[0]
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem.replace("-", " ").replace("_", " ").strip().title()
