"""OCR: reading the text out of scanned pages and pictures (alignment_plan RET-06).

Why: only text that is already text used to be indexed. A scanned PDF, a photographed page or a screenshot pasted into a Word file
produced an empty document that showed 'done' and could never be answered from.

How it fits in: `chunker.extract_document` calls this module for a PDF page that has a picture but (almost) no text, for pictures
embedded in a .docx, and for an uploaded image. The text that comes back goes through the SAME chunking, encryption, access tagging,
graph building and delete path as typed text; nothing downstream knows or cares that it came from a picture, except a `scanned`
flag used for the citation label.

The engine is LOCAL (Tesseract, Apache-2.0, installed in the Docker image). No page ever leaves the machine: a cloud OCR API would
break the self-hosted claim (ENG-06). The engine sits behind a two-method interface so another local engine can replace it.

Images are untrusted input, so every step has a limit (settings.py): pages per document, pixels per image, seconds per page.
A page the engine cannot read with enough confidence is REPORTED ('could not read page 3'), never silently indexed as noise.
"""
import io
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Optional

from settings import get_settings

log = logging.getLogger("mindsearch.ocr")

MIN_IMAGE_SIDE = 40          # pixels; smaller pictures are icons, bullets or logos, not pages of text


class OcrError(RuntimeError):
    """This page or picture could not be read (too large, timed out, damaged, engine failure). The caller reports it and moves on."""


@dataclass
class OcrText:
    text: str
    confidence: float        # 0-100: the engine's own average confidence over the words it found (0 when it found none)
    junk: int = 0            # leader-line junk tokens removed or trimmed (see clean_token)


# A dotted leader on a form ('Liquid Nitrogen .......... $200') is read by Tesseract as letter soup: 'ssccccecee', '.cccsees',
# 'services...........cceeeeeee'. Indexed, it is noise that searches can match. What tells it apart from words: real English
# words almost never repeat one letter three times in a row, and soup uses only a handful of different letters. Engine
# confidence does NOT tell them apart: measured on the DocVQA dev pages (evaluation.ocr_eval junk), every confidence cut-off
# that removed junk also removed real text that Tesseract scored 0 ('THE SMOOTH', 'Terrible', 'AGENDA'), and cut-offs on single
# words lost answers, because the dots are glued onto real words ('Good.ssccccecee').
_LEADER = re.compile(r"[.:,·•…_\-]{2,}")          # a run of dots / colons where a leader starts
_TRIPLE = re.compile(r"([a-z])\1\1")


def _is_soup(text: str) -> bool:
    letters = re.sub(r"[^a-z]", "", text.lower())
    return len(letters) >= 5 and len(set(letters)) <= 4 and bool(_TRIPLE.search(letters))


def clean_token(token: str) -> str:
    """The token without leader junk: 'Excellent....seseees' -> 'Excellent....', 'eseseeseeeeeneneese' -> '', 'Good' -> 'Good'.
    When the token has a dotted run, only the part after the dots is judged, so the real word before it always stays."""
    m = _LEADER.search(token)
    if m:
        tail = token[m.end():]
        return token[:m.end()] if tail and _is_soup(tail) else token
    return "" if _is_soup(token) else token


class TesseractEngine:
    """Tesseract through pytesseract. `read` returns the text in reading order with the mean word confidence."""
    name = "tesseract"

    def __init__(self, languages: str = "eng", timeout_s: float = 30.0):
        import pytesseract
        pytesseract.get_tesseract_version()          # raises if the tesseract program itself is not installed
        self._tess, self.languages, self.timeout_s = pytesseract, languages, timeout_s

    def word_table(self, image) -> dict:
        """Tesseract's raw word table (text, conf, block / paragraph / line numbers per word)."""
        try:
            return self._tess.image_to_data(image, lang=self.languages, timeout=self.timeout_s, output_type=self._tess.Output.DICT)
        except RuntimeError as e:                    # pytesseract raises RuntimeError when the time limit is hit
            raise OcrError(f"OCR timed out or failed ({type(e).__name__})") from e

    def read(self, image) -> OcrText:
        return words_to_text(self.word_table(image))


def words_to_text(data: dict, clean: bool = True) -> OcrText:
    """Rebuild lines and paragraphs from Tesseract's word table, and average the confidence of the real words.
    (Tesseract reports -1 for layout rows that are not words; those are ignored.) With `clean`, leader-line junk is removed
    (clean_token); the confidence still counts every word the engine found, so the page-level check is unchanged."""
    lines, confidences, last_par = {}, [], None
    order, junk = [], 0
    for i, word in enumerate(data.get("text", [])):
        word = (word or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not word or conf < 0:
            continue
        confidences.append(conf)
        if clean:
            kept = clean_token(word)
            if kept != word:
                junk += 1
            word = kept
            if not word:
                continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append(word)
    out = []
    for key in order:
        if last_par is not None and key[:2] != last_par:
            out.append("")                           # blank line between paragraphs, which the chunker treats as a break
        out.append(" ".join(lines[key]))
        last_par = key[:2]
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return OcrText("\n".join(out).strip(), confidence, junk)


# ------------------------------------------------------------------ engine selection

_engine = None
_engine_checked = False


def get_engine():
    """The OCR engine, or None when OCR is switched off (OCR_ENABLED=0) or Tesseract is not installed. Checked once per process;
    a missing engine is logged once and ingestion carries on without OCR (scanned pages are then reported as not read)."""
    global _engine, _engine_checked
    if not _engine_checked:
        _engine_checked = True
        cfg = get_settings()
        if cfg.ocr_enabled:
            try:
                _engine = TesseractEngine(cfg.ocr_languages, cfg.ocr_page_timeout_s)
            except Exception as e:  # noqa: BLE001 - library or program missing: run without OCR
                log.warning("OCR is unavailable (%s): scanned pages will be reported as not read", type(e).__name__)
    return _engine


def set_engine(engine):
    """Use this engine from now on (tests pass a fake; None switches OCR off). `reset_engine()` goes back to automatic."""
    global _engine, _engine_checked
    _engine, _engine_checked = engine, True


def reset_engine():
    global _engine, _engine_checked
    _engine, _engine_checked = None, False


# ------------------------------------------------------------------ images in, text out

def render_pdf_page(pdf_path: str, page_index: int, dpi: int, max_pixels: int):
    """Page `page_index` (0-based) of a PDF as a PIL image, at `dpi` unless that would exceed `max_pixels` (then it is scaled down:
    a poster-sized page must not allocate gigabytes)."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        page = pdf[page_index]
        width, height = page.get_size()              # in points (1/72 inch)
        scale = dpi / 72.0
        if width * height * scale * scale > max_pixels:
            scale = math.sqrt(max_pixels / (width * height))
        return page.render(scale=scale).to_pil()
    finally:
        pdf.close()


def open_image(data: bytes, max_pixels: int):
    """A PIL image from raw bytes, or None for a picture too small to hold text. Raises OcrError for a picture that is too large
    (checked from the header, BEFORE the pixels are decoded: decompression-bomb protection) or not a readable image."""
    from PIL import Image
    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        if width * height > max_pixels:
            raise OcrError(f"image too large ({width}x{height} pixels)")
        if min(width, height) < MIN_IMAGE_SIDE:
            return None
        image.load()
        return image.convert("RGB")
    except OcrError:
        raise
    except Exception as e:  # noqa: BLE001 - damaged file, unsupported format, PIL's own bomb check
        raise OcrError(f"image could not be opened ({type(e).__name__})") from e


def read_image(image, engine=None) -> Optional[OcrText]:
    """OCR one image. Returns the text only when the engine is confident enough (OCR_MIN_CONFIDENCE) and found something;
    otherwise None, which the caller reports as 'could not read'. Raises OcrError on a timeout or an engine failure."""
    from metrics import OCR_PAGES, OCR_SECONDS
    engine = engine or get_engine()
    if engine is None:
        raise OcrError("OCR is not available")
    started = time.perf_counter()
    try:
        result = engine.read(image)
    except OcrError:
        OCR_PAGES.labels("unreadable").inc()
        raise
    except Exception as e:  # noqa: BLE001 - an engine crash on one page must not fail the whole document
        OCR_PAGES.labels("unreadable").inc()
        raise OcrError(f"OCR failed ({type(e).__name__})") from e
    finally:
        OCR_SECONDS.observe(time.perf_counter() - started)
    if not result.text.strip() or result.confidence < get_settings().ocr_min_confidence:
        OCR_PAGES.labels("unreadable").inc()
        return None
    OCR_PAGES.labels("read").inc()
    return result
