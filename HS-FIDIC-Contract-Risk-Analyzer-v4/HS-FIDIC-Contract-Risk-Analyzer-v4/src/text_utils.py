"""Contract text extraction (PDF / DOCX / TXT) and sentence segmentation.

PDF text arrives as visual lines. Feeding raw lines to a classifier produces half-sentences, so
lines are re-joined into paragraphs, running headers/footers are removed, headings are kept as
their own units, and paragraphs are split into sentences while tracking the clause number.
"""
from __future__ import annotations

import io
import re
from collections import Counter

_SMALL = {"of", "and", "the", "to", "for", "in", "on", "by", "or", "a", "an", "with", "at", "under", "&", "-", "–"}
_MODAL = re.compile(r"\b(shall|may|must|will|is|are|be|has|have|means|was|were)\b", re.I)
_DANGLING = re.compile(r"\b(the|a|an|of|to|for|and|or|in|on|by|with|be|any|such|this|that|which|shall|may|its|his|their|from|under|as)\s*$", re.I)
_CLAUSE_NO = re.compile(r"^\s*((?:Clause|Sub-Clause|Article)\s+)?(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+(?=\S)", re.I)
_PAGE_NO = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.I)
_ABBREV = re.compile(r"\b(e\.g|i\.e|etc|No|no|Nos|Art|Cl|para|Ltd|Inc|Co|vs|approx|Mr|Ms|Dr)\.", re.U)
_TOC = re.compile(r"(\.\s?){5,}\s*\d*\s*$|…{2,}|_{5,}")  # table-of-contents dot leaders / form blanks
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=(?:\(?[a-z0-9ivx]{1,4}\)\s+)?[A-Z“\"(])")


def is_heading_like(s: str) -> bool:
    """Strict structural heading test, used for label correction and segmentation."""
    s = str(s).strip()
    if not s or len(s) > 110:
        return False
    body = _CLAUSE_NO.sub("", s).strip()
    words = body.split()
    if not 1 <= len(words) <= 9:
        return False
    if s[-1] in ".;:,":
        return False
    if ", " in s and not _CLAUSE_NO.match(s):
        return False
    if _MODAL.search(body):
        return False
    if not (s[0].isupper() or s[0].isdigit()):
        return False
    sig = [w for w in words if w.lower() not in _SMALL]
    if not sig:
        return False
    caps = sum(1 for w in sig if w[:1].isupper() or w[:1] in "“\"'([" or w[:1].isdigit())
    return caps / len(sig) >= 0.75


def is_fragment(s: str) -> bool:
    """Mid-sentence PDF line fragments (start lower-case or end on a dangling word / comma)."""
    s = str(s).strip()
    if not s:
        return True
    if s[0].islower():
        return True
    if s.endswith(",") or _DANGLING.search(s):
        return True
    return False


def _clean(s: str) -> str:
    s = s.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = re.sub(r"(\w)-\s+(\w)", lambda m: m.group(1) + m.group(2) if m.group(2).islower() else m.group(0), s)
    return re.sub(r"\s+", " ", s).strip()


def _repeated_lines(pages):
    """Lines that occur on >=40% of pages (running headers / footers)."""
    if len(pages) < 3:
        return set()
    c = Counter()
    for _, text in pages:
        seen = {re.sub(r"\d+", "#", l.strip()) for l in (text or "").splitlines() if l.strip()}
        c.update(seen)
    return {l for l, n in c.items() if n >= max(3, 0.4 * len(pages)) and len(l) < 120}


def _split_sentences(paragraph: str):
    p = _ABBREV.sub(lambda m: m.group(0).replace(".", "§DOT§"), paragraph)
    p = re.sub(r"(\d)\.(\d)", r"\1§DOT§\2", p)
    parts = [x.replace("§DOT§", ".").strip() for x in _SENT_SPLIT.split(p)]
    out = []
    for x in parts:
        if not x:
            continue
        if len(x.split()) > 140:  # very long enumerations -> split at list items
            out.extend(y.strip() for y in re.split(r";\s+(?=\([a-z0-9ivx]{1,4}\))", x) if y.strip())
        else:
            out.append(x)
    merged = []
    for x in out:
        if merged and (len(x) < 15 or x[0].islower()) and not is_heading_like(x):
            merged[-1] = merged[-1] + " " + x
        else:
            merged.append(x)
    return merged


def segment_pages(pages):
    """pages: iterable of (page_number, text) -> list of {page, clause, sentence, is_heading}."""
    pages = list(pages)
    drop = _repeated_lines(pages)
    rows, buf, buf_page = [], [], None
    clause = ""
    head = {"top": "", "sub": ""}

    def flush():
        nonlocal buf, buf_page
        if buf:
            para = _clean(" ".join(buf))
            for s in _split_sentences(para):
                if len(re.findall(r"[A-Za-z]{2,}", s)) >= 2:
                    rows.append({"page": buf_page, "clause": clause, "sentence": s, "is_heading": False,
                                 "heading": " | ".join(x for x in (head["top"], head["sub"]) if x)})
        buf, buf_page = [], None

    for page_no, text in pages:
        for raw in (text or "").replace("\r", "\n").split("\n"):
            line = raw.strip()
            if not line:
                if buf and re.search(r"[.;:]$", buf[-1]):
                    flush()
                continue
            if _PAGE_NO.match(line) or re.sub(r"\d+", "#", line) in drop or _TOC.search(line):
                continue
            line = _clean(line)
            m = _CLAUSE_NO.match(line)
            starts_clause = bool(m) and (len(m.group(2)) > 1 or m.group(1))
            if is_heading_like(line) and (not buf or re.search(r"[.;:]$", buf[-1]) or starts_clause):
                flush()
                if m:
                    clause = m.group(2)
                    if "." in clause:
                        head["sub"] = line
                    else:
                        head["top"], head["sub"] = line, ""
                else:
                    head["sub"] = line
                rows.append({"page": page_no, "clause": clause, "sentence": line, "is_heading": True,
                             "heading": head["top"]})
                continue
            if starts_clause:
                flush()
                clause = m.group(2)
            if not buf:
                buf_page = page_no
            buf.append(line)
            if re.search(r"[.!?]$", line) and len(line) < 60:
                # short line ending a sentence often closes a paragraph in PDFs
                pass
    flush()
    return rows


def extract_uploaded_file(name: str, raw: bytes):
    name = name.lower()
    if name.endswith(".pdf"):
        pages = []
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except Exception:
            pages = []
        total = sum(len(t) for _, t in pages)
        if total < 200:
            try:
                import pdfplumber

                with pdfplumber.open(io.BytesIO(raw)) as pdf:
                    pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(pdf.pages)]
                total = sum(len(t) for _, t in pages)
            except Exception:
                pass
        return pages, {"kind": "pdf", "pages": len(pages), "text_chars": total,
                       "ocr_warning": total < max(300, 25 * max(len(pages), 1))}
    if name.endswith(".docx"):
        from docx import Document

        doc = Document(io.BytesIO(raw))
        text = "\n".join(p.text for p in doc.paragraphs)
        for t in doc.tables:
            for r in t.rows:
                text += "\n" + " ".join(c.text for c in r.cells)
        return [(1, text)], {"kind": "docx", "pages": 1, "text_chars": len(text), "ocr_warning": len(text) < 300}
    if name.endswith(".txt"):
        text = raw.decode("utf-8", errors="replace")
        return [(1, text)], {"kind": "txt", "pages": 1, "text_chars": len(text), "ocr_warning": len(text) < 300}
    raise ValueError("Unsupported file type. Upload PDF, DOCX or TXT.")
