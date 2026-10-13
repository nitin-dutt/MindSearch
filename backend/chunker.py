"""Turn an uploaded file into passages ("chunks") for search (alignment_plan RET-01).

Three things matter for retrieval quality and are handled here:
  * PDFs hard-wrap every visual line, so each line used to become its own "paragraph" and chunks broke mid-sentence. PDF lines are now
    re-flowed into real paragraphs (and words split by a line-end hyphen are re-joined).
  * A paragraph longer than the chunk size used to become one oversized chunk, whose tail the embedding model never saw. Long paragraphs
    are now split at sentence boundaries (`split_long=True`, the default).
  * Every chunk remembers its page range for PDFs, so answers can cite "page 3".
Plain text, Markdown and DOCX keep their line-per-paragraph behaviour. The evaluation corpora (one passage per line) must stay whole, so the
evaluation harness runs with `split_long=False`: its chunks are identical to the original chunker's.
"""
import os
import re
import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple

import chardet
import nltk
import pdfplumber
from docx import Document

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize  # noqa: E402  (after the data check on purpose)


@dataclass
class Chunk:
    text: str
    page_start: Optional[int] = None      # 1-based page numbers (PDF only)
    page_end: Optional[int] = None


# ------------------------------------------------------------------ extraction

def extract_pages(file_path) -> List[Tuple[Optional[int], str]]:
    """[(page number or None, text)]. PDFs give one entry per page; other formats one entry with page None."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    try:
        if ext == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                return [(i, page.extract_text() or "") for i, page in enumerate(pdf.pages, start=1)]
        if ext == '.docx':
            return [(None, extract_text_from_docx(file_path))]
        return [(None, extract_text_from_txt(file_path))]
    except Exception as e:
        print(f"Error extracting text from {file_path}: {str(e)}")
        return []


def extract_text_from_file(file_path):
    """All text of a file as one string (PDF pages separated by a blank line, never glued together)."""
    return "\n\n".join(text for _, text in extract_pages(file_path))


def extract_text_from_docx(file_path):
    """Extract text from DOCX files"""
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX: {str(e)}")
        return ""


def extract_text_from_txt(file_path):
    """Extract text from TXT files with encoding detection"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'rb') as f:
                raw = f.read()
                result = chardet.detect(raw)
                encoding = result.get('encoding') or 'utf-8'
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            print(f"Error detecting encoding: {str(e)}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()


# ------------------------------------------------------------------ paragraphs

_ITEM_START = re.compile(r"^(•|[-*–]\s|\d+[.)]\s|\(?[a-zA-Z][.)]\s)")


def reflow_pdf_page(text: str) -> List[str]:
    """Rebuild paragraphs from a PDF page whose lines are hard-wrapped at the page width.

    A paragraph break is only placed where it cannot cut a sentence: after a line that ends a sentence (. ! ? : ;) when the next line starts
    with a capital letter or digit, before list items, after a clearly short heading-like line, and at blank lines. Everything else is
    one wrapped sentence and is joined. (A paragraph may therefore be split at a sentence boundary; the packer puts those pieces back
    into the same chunk.) A word split by a hyphen at the end of a line is re-joined."""
    lines = [ln.strip() for ln in text.splitlines()]
    lengths = [len(ln) for ln in lines if ln]
    typical = statistics.median(lengths) if lengths else 0
    paragraphs, current = [], []

    def flush():
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    for line in lines:
        if not line:
            flush()
            continue
        if current:
            prev = current[-1]
            heading_like = len(prev) < 0.5 * typical and line[0].isupper()
            ends_sentence = prev.endswith(('.', '!', '?', ':', ';'))
            if _ITEM_START.match(line) or heading_like or (ends_sentence and (line[0].isupper() or line[0].isdigit())):
                flush()
        if current and current[-1].endswith('-') and len(current[-1]) > 1 and current[-1][-2].isalpha() and line[:1].islower():
            current[-1] = current[-1][:-1] + line            # "informa-" + "tion" -> "information"
        else:
            current.append(line)
    flush()
    return paragraphs


def split_long_paragraph(paragraph: str, max_words: int) -> List[str]:
    """Pieces of at most `max_words` words, cut at sentence boundaries (a single over-long sentence is cut by words)."""
    if len(paragraph.split()) <= max_words:
        return [paragraph]
    pieces, current, count = [], [], 0
    for sentence in sent_tokenize(paragraph):
        words = sentence.split()
        if len(words) > max_words:                           # one enormous sentence: cut by words
            if current:
                pieces.append(" ".join(current))
                current, count = [], 0
            pieces.extend(" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words))
            continue
        if count + len(words) > max_words and current:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += len(words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _paragraphs(file_path, chunk_size, split_long):
    """[(paragraph text, page)] in reading order."""
    is_pdf = os.path.splitext(file_path)[1].lower() == '.pdf'
    out = []
    for page, text in extract_pages(file_path):
        raw = reflow_pdf_page(text) if is_pdf else [p.strip() for p in text.split('\n') if p.strip()]
        for para in raw:
            for piece in (split_long_paragraph(para, chunk_size) if split_long else [para]):
                out.append((piece, page))
    return out


# ------------------------------------------------------------------ chunking

def run_chunking_with_pages(file_path, chunk_size=150, overlap=20, split_long=True) -> List[Chunk]:
    """Pack paragraphs into chunks of about `chunk_size` words, overlapping by about `overlap` words, remembering the page range."""
    paragraphs = _paragraphs(file_path, chunk_size, split_long)
    if not paragraphs:
        return []

    chunks: List[Chunk] = []
    current: List[Tuple[str, Optional[int]]] = []
    current_length = 0

    def emit():
        pages = [p for _, p in current if p is not None]
        chunks.append(Chunk("\n".join(t for t, _ in current), min(pages) if pages else None, max(pages) if pages else None))

    for para, page in paragraphs:
        para_len = len(para.split())
        if current_length + para_len > chunk_size and current:
            emit()
            overlap_part, overlap_length = [], 0
            for prev in reversed(current):                   # keep the last few paragraphs as the overlap
                n = len(prev[0].split())
                if overlap_length + n <= overlap:
                    overlap_part.insert(0, prev)
                    overlap_length += n
                else:
                    break
            current, current_length = overlap_part, overlap_length
        current.append((para, page))
        current_length += para_len
    if current:
        emit()
    return chunks


def run_chunking(file_path, chunk_size=150, overlap=20, split_long=True):
    """The chunk texts only (see run_chunking_with_pages for page numbers)."""
    return [c.text for c in run_chunking_with_pages(file_path, chunk_size, overlap, split_long)]
