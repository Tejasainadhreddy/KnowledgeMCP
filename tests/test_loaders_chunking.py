import pytest
from conftest import SAMPLE_DOCS

from knowledgemcp.chunking import chunk_sections, split_text
from knowledgemcp.loaders import UnsupportedFileType, load_document, load_markdown, title_from


def test_markdown_sections_keep_heading_path():
    sections = load_markdown("# Guide\nintro\n## Setup\nstep one\n### Linux\napt install\n## Usage\nrun it")
    assert [s.section for s in sections] == ["Guide", "Guide > Setup", "Guide > Setup > Linux", "Guide > Usage"]
    assert sections[2].text == "apt install"


def test_markdown_ignores_hash_lines_inside_code_blocks():
    sections = load_markdown("# A\n```bash\n# not a heading\n```\ntext")
    assert len(sections) == 1 and "# not a heading" in sections[0].text


def test_pdf_pages_are_numbered():
    sections = load_document("finance/expense-policy.pdf", (SAMPLE_DOCS / "finance/expense-policy.pdf").read_bytes())
    assert [s.page for s in sections] == [1, 2, 3]
    assert "1,000 US dollars" in sections[1].text


def test_text_file_and_unsupported_type():
    assert load_document("public/a.txt", b"hello")[0].text == "hello"
    with pytest.raises(UnsupportedFileType):
        load_document("x.docx", b"...")


def test_title_from_h1_or_filename():
    assert title_from("hr/pto.md", load_markdown("# Paid Time Off\nx")) == "Paid Time Off"
    assert title_from("public/company-overview.txt", []) == "Company Overview"


def test_split_text_respects_limit_and_overlap():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(10))
    chunks = split_text(text, max_chars=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)
    assert any(c.split()[0] != "Paragraph" for c in chunks[1:])  # later chunks start with overlap text


def test_chunks_never_cross_sections():
    sections = load_markdown("# A\n" + "alpha " * 300 + "\n# B\nbeta")
    chunks = chunk_sections("d1", "hr", sections, max_chars=400, overlap=50)
    assert {c.section for c in chunks} == {"A", "B"}
    assert chunks[-1].text.startswith("[B]") and chunks[-1].chunk_id == f"d1:{len(chunks) - 1}"
