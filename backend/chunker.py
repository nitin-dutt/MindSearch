"""Turn an uploaded file into passages ("chunks") for search (alignment_plan RET-01).

Three things matter for retrieval quality and are handled here:
  * PDFs hard-wrap every visual line, so each line used to become its own "paragraph" and chunks broke mid-sentence. PDF lines are now
    re-flowed into real paragraphs (and words split by a line-end hyphen are re-joined).
  * A paragraph longer than the chunk size used to become one oversized chunk, whose tail the embedding model never saw. Long paragraphs
    are now split at sentence boundaries (`split_long=True`, the default).
  * Every chunk remembers its page range for PDFs, so answers can cite "page 3".
Plain text, Markdown and DOCX keep their line-per-paragraph behaviour. The evaluation corpora (one passage per line) must stay whole, so the
evaluation harness runs with `split_long=False`: its chunks are identical to the original chunker's.

Scanned pages and pictures (alignment_plan RET-06): a PDF page that holds a picture but almost no text, a picture embedded in a .docx
and an uploaded image are read with a local OCR engine (ocr.py). Their text is chunked like any other text; the chunk is flagged `ocr`
so the citation can say "page 3 (scanned)". Pages the engine could not read are listed in the ExtractionReport, never indexed.
"""
import logging
import os
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import chardet
import nltk
import pdfplumber
from docx import Document

import ocr
from settings import get_settings

log = logging.getLogger("mindsearch.chunker")

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
    ocr: bool = False                     # some of this text was read from a scanned page or a picture


@dataclass
class PageText:
    page: Optional[int]                   # 1-based PDF page, None for formats without pages
    text: str
    ocr: bool = False


MIN_ENCODING_CONFIDENCE = 0.5   # chardet confidence below which a non-UTF-8 text file is refused instead of guessed


class DocumentReadError(ValueError):
    """The file could not be read at all (damaged, mislabelled, undecodable). The message is safe to show the uploader; the
    document is marked failed with it and not retried, because the same file fails the same way every time."""


@dataclass
class ExtractionReport:
    """What happened to the scanned pages and pictures of one document. Stored with the document so the user is told
    'could not read page 3' instead of finding out when a question goes unanswered."""
    ocr_pages: int = 0                                           # pages / pictures read by OCR and indexed
    unreadable_pages: List[int] = field(default_factory=list)    # PDF pages OCR could not read (low confidence, nothing found, timeout)
    unreadable_images: int = 0                                   # pictures (in a .docx, or an uploaded image) OCR could not read
    skipped: int = 0                                             # scanned pages / pictures not attempted: OCR off or over OCR_MAX_PAGES
    junk_removed: int = 0                                        # leader-line junk tokens removed from OCR text (ocr.clean_token)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def attempted(self) -> int:
        return self.ocr_pages + len(self.unreadable_pages) + self.unreadable_images


# ------------------------------------------------------------------ extraction

def no_text_message(scan: dict) -> str:
    """Why a document produced no passages, in words a user can act on (`scan` is ExtractionReport.to_dict())."""
    if scan.get("unreadable_pages") or scan.get("unreadable_images"):
        return "No readable text: the scanned pages or pictures could not be read (too blurry, handwritten, or not text)."
    if scan.get("skipped"):
        return "No readable text: this file is scanned and OCR is not available on this server."
    return "No readable text was found in this file."


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def extract_document(file_path) -> Tuple[List[PageText], ExtractionReport]:
    """The text of a file as pages, with OCR for the parts that are pictures, and a report of what OCR could and could not read.
    PDFs give one entry per page; a .docx gives its typed text and then one entry per readable picture; an image gives one entry."""
    ext = os.path.splitext(file_path)[1].lower()
    report = ExtractionReport()
    try:
        if ext == '.pdf':
            return _extract_pdf(file_path, report), report
        if ext == '.docx':
            return [PageText(None, extract_text_from_docx(file_path))] + _docx_pictures(file_path, report), report
        if ext in IMAGE_EXTENSIONS:
            with open(file_path, 'rb') as f:
                text = _ocr_picture(f.read(), report)
            return ([PageText(None, text, ocr=True)] if text else []), report
        return [PageText(None, extract_text_from_txt(file_path))], report
    except DocumentReadError:
        raise
    except Exception as e:  # noqa: BLE001 - a damaged or mislabelled file: say so, never pretend it was empty
        log.error("could not read %s: %s: %s", os.path.basename(file_path), type(e).__name__, e)
        kind = ext.lstrip(".").upper() or "this"
        raise DocumentReadError(f"The {kind} file could not be read ({type(e).__name__}); it may be damaged or not really a "
                                f"{kind} file.") from e


def extract_pages(file_path) -> List[Tuple[Optional[int], str]]:
    """[(page number or None, text)]. PDFs give one entry per page; other formats one entry with page None."""
    return [(p.page, p.text) for p in extract_document(file_path)[0]]


def _ocr_metric(result: str):
    from metrics import OCR_PAGES
    OCR_PAGES.labels(result).inc()


def _can_ocr(report: ExtractionReport) -> bool:
    """True if one more page may be sent to OCR; otherwise the page is counted as skipped."""
    if ocr.get_engine() is None or report.attempted >= get_settings().ocr_max_pages:
        report.skipped += 1
        _ocr_metric("skipped")
        return False
    return True


def _extract_pdf(file_path, report: ExtractionReport) -> List[PageText]:
    cfg = get_settings()
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # A scanned page is a picture of text: (almost) nothing to extract, but an image on the page. A page with neither is
            # simply blank and is left alone.
            if len(text.strip()) < cfg.ocr_min_chars and page.images and _can_ocr(report):
                try:
                    read = ocr.read_image(ocr.render_pdf_page(file_path, number - 1, cfg.ocr_dpi, cfg.ocr_max_pixels))
                except Exception:  # noqa: BLE001 - OcrError, or a page the renderer cannot draw: report it, keep going
                    read = None
                if read is None:
                    report.unreadable_pages.append(number)
                else:
                    report.ocr_pages += 1
                    report.junk_removed += read.junk
                    pages.append(PageText(number, read.text, ocr=True))
                    continue
            pages.append(PageText(number, text))
    return pages


def _ocr_picture(data: bytes, report: ExtractionReport) -> Optional[str]:
    """Text of one picture, or None (too small to hold text, not attempted, or not readable: the last is counted in the report)."""
    cfg = get_settings()
    try:
        image = ocr.open_image(data, cfg.ocr_max_pixels)
    except ocr.OcrError:
        report.unreadable_images += 1
        _ocr_metric("unreadable")
        return None
    if image is None or not _can_ocr(report):
        return None
    try:
        read = ocr.read_image(image)
    except ocr.OcrError:
        read = None
    if read is None:
        report.unreadable_images += 1
        return None
    report.ocr_pages += 1
    report.junk_removed += read.junk
    return read.text


def _docx_pictures(file_path, report: ExtractionReport) -> List[PageText]:
    """OCR text of the pictures embedded in a Word file (screenshots, scanned pages, tables saved as images), in the order Word
    stores them. They are added after the typed text: a .docx does not say on which page a picture sits."""
    out = []
    for rel in Document(file_path).part.rels.values():
        if rel.is_external or not rel.reltype.endswith("/image"):
            continue
        text = _ocr_picture(rel.target_part.blob, report)
        if text:
            out.append(PageText(None, text, ocr=True))
    return out


def extract_text_from_file(file_path):
    """All text of a file as one string (PDF pages separated by a blank line, never glued together)."""
    return "\n\n".join(text for _, text in extract_pages(file_path))


def extract_text_from_docx(file_path):
    """Text of a .docx file. A damaged file raises (extract_document turns that into DocumentReadError)."""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_txt(file_path):
    """Text of a plain-text file: UTF-8, or the encoding chardet detects. If neither decodes it, DocumentReadError. The file is
    never read with characters silently dropped, because the index would then hold text the file does not say."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        pass
    with open(file_path, 'rb') as f:
        raw = f.read()
    guess = chardet.detect(raw)
    encoding, confidence = guess.get('encoding'), guess.get('confidence') or 0.0
    # A low-confidence guess decodes WITHOUT any error into the wrong characters (measured: Windows-1252 text guessed as cp850 at
    # 11%), which would put text the file does not say into the index. Below this confidence the upload fails with the reason.
    if not encoding or confidence < MIN_ENCODING_CONFIDENCE:
        best = f" (best guess {encoding} at {confidence:.0%})" if encoding else ""
        raise DocumentReadError(f"The text encoding of this file could not be detected reliably{best}. Save it as UTF-8 and upload "
                                "it again.")
    # Like web browsers (WHATWG encoding standard), read an ISO-8859-1 / ASCII guess as Windows-1252: it is a superset, and real
    # files with that label are almost always Windows-1252 (otherwise '€', curly quotes and dashes become invisible control codes).
    if encoding.lower().replace("_", "-") in ("iso-8859-1", "latin-1", "ascii"):
        encoding = "cp1252"
    try:
        return raw.decode(encoding)
    except (UnicodeDecodeError, LookupError) as e:
        raise DocumentReadError(f"The text could not be decoded (the detected encoding, {encoding}, did not fit). Save it as UTF-8 "
                                "and upload it again.") from e


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
    """([(paragraph text, page, read by OCR)] in reading order, the extraction report)."""
    is_pdf = os.path.splitext(file_path)[1].lower() == '.pdf'
    out = []
    pages, report = extract_document(file_path)
    for p in pages:
        # OCR text is hard-wrapped at the line ends of the picture, exactly like a PDF page, so it is re-flowed the same way
        raw = reflow_pdf_page(p.text) if (is_pdf or p.ocr) else [x.strip() for x in p.text.split('\n') if x.strip()]
        for para in raw:
            for piece in (split_long_paragraph(para, chunk_size) if split_long else [para]):
                out.append((piece, p.page, p.ocr))
    return out, report


# ------------------------------------------------------------------ chunking

def run_chunking_with_pages(file_path, chunk_size=150, overlap=20, split_long=True) -> List[Chunk]:
    """Pack paragraphs into chunks of about `chunk_size` words, overlapping by about `overlap` words, remembering the page range."""
    return chunk_document(file_path, chunk_size, overlap, split_long)[0]


def chunk_document(file_path, chunk_size=150, overlap=20, split_long=True) -> Tuple[List[Chunk], ExtractionReport]:
    """(chunks, extraction report): run_chunking_with_pages plus what OCR could and could not read."""
    paragraphs, report = _paragraphs(file_path, chunk_size, split_long)
    if not paragraphs:
        return [], report

    chunks: List[Chunk] = []
    current: List[Tuple[str, Optional[int], bool]] = []
    current_length = 0

    def emit():
        pages = [p for _, p, _ in current if p is not None]
        chunks.append(Chunk("\n".join(t for t, _, _ in current), min(pages) if pages else None, max(pages) if pages else None,
                            ocr=any(scanned for _, _, scanned in current)))

    for para, page, scanned in paragraphs:
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
        current.append((para, page, scanned))
        current_length += para_len
    if current:
        emit()
    return chunks, report


def run_chunking(file_path, chunk_size=150, overlap=20, split_long=True):
    """The chunk texts only (see run_chunking_with_pages for page numbers)."""
    return [c.text for c in run_chunking_with_pages(file_path, chunk_size, overlap, split_long)]
