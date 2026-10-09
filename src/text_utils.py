from __future__ import annotations
import io
import re
from pathlib import Path

_HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+){0,4}|[A-Z])\s+.{2,90}$")
_CLAUSE_PREFIX_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,5})\s+")
_BULLET_SPLIT_RE = re.compile(r"(?=\s+\([a-zA-Z0-9ivxIVX]+\)\s+)")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'])")

def _clean(s: str) -> str:
    s = s.replace("\u00ad", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _split_paragraph(paragraph: str):
    p = _clean(paragraph)
    if not p:
        return []
    if _HEADING_RE.match(p) and len(p.split()) <= 14:
        return [p]

    chunks = []
    for bullet_chunk in _BULLET_SPLIT_RE.split(p):
        bullet_chunk = _clean(bullet_chunk)
        if not bullet_chunk:
            continue
        parts = _SENTENCE_SPLIT_RE.split(bullet_chunk)
        chunks.extend(_clean(x) for x in parts if _clean(x))

    # Merge very short fragments with the preceding sentence.
    merged = []
    for c in chunks:
        if len(c) < 20 and merged and not _HEADING_RE.match(c):
            merged[-1] = _clean(merged[-1] + " " + c)
        else:
            merged.append(c)
    return merged

def segment_pages(pages):
    """
    pages: iterable of (page_number, text)
    returns rows with page/clause/sentence.
    """
    rows = []
    current_clause = ""
    for page_number, text in pages:
        text = text or ""
        text = text.replace("\r", "\n")
        # Keep meaningful line/paragraph boundaries.
        paras = [x.strip() for x in re.split(r"\n{1,}", text) if x.strip()]
        for para in paras:
            para = _clean(para)
            m = _CLAUSE_PREFIX_RE.match(para)
            if m:
                current_clause = m.group(1)
            for sentence in _split_paragraph(para):
                cm = _CLAUSE_PREFIX_RE.match(sentence)
                if cm:
                    current_clause = cm.group(1)
                rows.append({
                    "page": page_number,
                    "clause": current_clause,
                    "sentence": sentence,
                })
    return rows

def extract_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        total_chars = sum(len(t) for _, t in pages)
        return pages, {
            "kind": "pdf",
            "pages": len(pages),
            "text_chars": total_chars,
            "ocr_warning": total_chars < max(300, 25 * len(pages)),
        }

    if name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(raw))
        text = "\n".join(p.text for p in doc.paragraphs)
        return [(1, text)], {"kind": "docx", "pages": None, "text_chars": len(text), "ocr_warning": False}

    if name.endswith(".txt"):
        text = raw.decode("utf-8", errors="replace")
        return [(1, text)], {"kind": "txt", "pages": None, "text_chars": len(text), "ocr_warning": False}

    raise ValueError("Unsupported file type. Upload PDF, DOCX, or TXT.")
