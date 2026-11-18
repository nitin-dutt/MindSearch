"""OCR for scanned pages and pictures (alignment_plan RET-06).

The OCR engine is replaced by a fake that returns prepared text, so these tests need no Tesseract and check OUR logic: which pages are
sent to OCR, what is indexed, what is reported as unreadable, and the limits on untrusted images. One test at the end runs the real
engine and is skipped where Tesseract is not installed.
"""
import io

import pytest

pytest.importorskip("PIL")
pytest.importorskip("pypdfium2")

from PIL import Image, ImageDraw  # noqa: E402

import chunker as C  # noqa: E402
import ocr  # noqa: E402
import uploads  # noqa: E402
from test_chunker import make_pdf  # noqa: E402

SENTENCE = "The warehouse lease in Rotterdam was signed on the fourth of March and runs for nine years."


class FakeEngine:
    """Returns the prepared results in order and remembers the size of every image it was shown."""
    name = "fake"

    def __init__(self, *results):
        self.results, self.seen = list(results), []

    def read(self, image):
        self.seen.append(image.size)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def good(text=SENTENCE, confidence=91.0):
    return ocr.OcrText(text, confidence)


@pytest.fixture
def engine():
    """Install a fake engine for one test; `engine(...)` sets what it will return."""
    def install(*results):
        fake = FakeEngine(*results)
        ocr.set_engine(fake)
        return fake
    yield install
    ocr.reset_engine()


def picture(text="scan", size=(400, 200)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((10, 10), text, fill="black")
    return img


def png_bytes(size=(400, 200)) -> bytes:
    buf = io.BytesIO()
    picture(size=size).save(buf, format="PNG")
    return buf.getvalue()


def scanned_pdf(path, pages=1):
    """A PDF whose pages are pictures with no text layer: what a scanner produces."""
    imgs = [picture(f"page {i}") for i in range(1, pages + 1)]
    imgs[0].save(path, format="PDF", save_all=True, append_images=imgs[1:])
    return str(path)


# ------------------------------------------------------------------ which pages are read, what is indexed, what is reported

def test_a_scanned_pdf_page_is_read_indexed_with_its_page_number_and_flagged(tmp_path, engine):
    fake = engine(good())
    chunks, report = C.chunk_document(scanned_pdf(tmp_path / "scan.pdf"))
    assert len(fake.seen) == 1
    assert [c.text for c in chunks] == [SENTENCE] and chunks[0].ocr is True and chunks[0].page_start == 1
    assert report.to_dict() == {"ocr_pages": 1, "unreadable_pages": [], "unreadable_images": 0, "skipped": 0, "junk_removed": 0}


def test_a_low_confidence_page_is_reported_and_never_indexed(tmp_path, engine):
    engine(good(), ocr.OcrText("x7 ~~ lI| 0O", 31.0), ocr.OcrText("", 0.0))
    chunks, report = C.chunk_document(scanned_pdf(tmp_path / "scan.pdf", pages=3))
    assert [c.text for c in chunks] == [SENTENCE]                       # the noise of page 2 is not searchable
    assert report.ocr_pages == 1 and report.unreadable_pages == [2, 3]  # low confidence, and a page where nothing was found


def test_a_page_that_already_has_text_is_not_sent_to_ocr_and_a_blank_page_is_left_alone(tmp_path, engine):
    fake = engine()
    pdf = tmp_path / "typed.pdf"
    make_pdf(pdf, [["This typed page has plenty of real text on it already."], []])     # page 2 is blank: no text, no picture
    chunks, report = C.chunk_document(str(pdf))
    assert fake.seen == [] and all(not c.ocr for c in chunks) and len(chunks) == 1
    assert report.attempted == 0 and report.skipped == 0


def test_a_timeout_or_engine_crash_on_one_page_does_not_lose_the_others(tmp_path, engine):
    engine(ocr.OcrError("OCR timed out"), ValueError("engine bug"), good())
    chunks, report = C.chunk_document(scanned_pdf(tmp_path / "scan.pdf", pages=3))
    assert [c.page_start for c in chunks] == [3] and report.unreadable_pages == [1, 2]


def test_without_an_engine_scanned_pages_are_counted_as_skipped(tmp_path):
    ocr.set_engine(None)
    try:
        chunks, report = C.chunk_document(scanned_pdf(tmp_path / "scan.pdf", pages=2))
    finally:
        ocr.reset_engine()
    assert chunks == [] and report.skipped == 2 and report.attempted == 0
    assert "OCR is not available" in C.no_text_message(report.to_dict())


def test_the_per_document_page_limit_stops_ocr_and_counts_the_rest(tmp_path, engine, monkeypatch):
    monkeypatch.setenv("OCR_MAX_PAGES", "2")
    fake = engine(good("First page text is long enough to count as a real sentence here."), good())
    chunks, report = C.chunk_document(scanned_pdf(tmp_path / "scan.pdf", pages=5))
    assert len(fake.seen) == 2 and report.ocr_pages == 2 and report.skipped == 3


# ------------------------------------------------------------------ untrusted images: size limits

def test_a_huge_page_is_rendered_scaled_down_to_the_pixel_limit(tmp_path, engine, monkeypatch):
    monkeypatch.setenv("OCR_MAX_PIXELS", "50000")
    fake = engine(good())
    C.chunk_document(scanned_pdf(tmp_path / "scan.pdf"))
    width, height = fake.seen[0]
    assert width * height <= 50000 * 1.02                                # never more pixels than allowed (rounding aside)


def test_an_oversize_picture_is_refused_before_it_is_decoded_and_reported(tmp_path, engine, monkeypatch):
    monkeypatch.setenv("OCR_MAX_PIXELS", "10000")
    fake = engine()
    img = tmp_path / "big.png"
    img.write_bytes(png_bytes(size=(400, 200)))                          # 80,000 pixels > 10,000
    chunks, report = C.chunk_document(str(img))
    assert chunks == [] and fake.seen == [] and report.unreadable_images == 1
    with pytest.raises(ocr.OcrError):
        ocr.open_image(png_bytes(size=(400, 200)), max_pixels=10000)


def test_a_damaged_image_is_reported_not_raised(tmp_path, engine):
    engine()
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n" + b"not really a picture")
    chunks, report = C.chunk_document(str(bad))
    assert chunks == [] and report.unreadable_images == 1
    assert "could not be read" in C.no_text_message(report.to_dict())


# ------------------------------------------------------------------ pictures in Word files, and uploaded images

def test_pictures_embedded_in_a_word_file_are_read_and_tiny_icons_are_ignored(tmp_path, engine):
    docx = pytest.importorskip("docx")
    fake = engine(good("Table saved as a picture: Rotterdam warehouse rent is 4,200 euros per month in total."))
    doc = docx.Document()
    doc.add_paragraph("Typed paragraph about the lease.")
    doc.add_picture(io.BytesIO(png_bytes(size=(400, 200))))
    doc.add_picture(io.BytesIO(png_bytes(size=(20, 20))))                # an icon: too small to hold text
    path = tmp_path / "with_picture.docx"
    doc.save(path)
    chunks, report = C.chunk_document(str(path))
    text = "\n".join(c.text for c in chunks)
    assert "Typed paragraph about the lease." in text and "4,200 euros" in text
    assert len(fake.seen) == 1 and report.ocr_pages == 1 and any(c.ocr for c in chunks)


def test_an_uploaded_image_is_read_as_one_scanned_passage(tmp_path, engine):
    engine(good())
    img = tmp_path / "photo.png"
    img.write_bytes(png_bytes())
    chunks, report = C.chunk_document(str(img))
    assert [c.text for c in chunks] == [SENTENCE] and chunks[0].ocr and chunks[0].page_start is None and report.ocr_pages == 1


def test_image_uploads_are_allowed_only_with_a_real_image_signature(tmp_path):
    path = uploads.safe_upload_path(str(tmp_path), "holiday photo.PNG")
    assert path.endswith(".png")
    uploads.validate_content(path, png_bytes())
    with pytest.raises(uploads.UploadError):
        uploads.validate_content(path, b"MZ this is a program, not a picture")
    jpg = uploads.safe_upload_path(str(tmp_path), "scan.jpeg")
    with pytest.raises(uploads.UploadError):
        uploads.validate_content(jpg, png_bytes())                       # a PNG renamed to .jpeg


# ------------------------------------------------------------------ the engine's word table -> text

def test_words_are_rebuilt_into_lines_and_paragraphs_and_layout_rows_do_not_count():
    data = {"text": ["", "The", "lease", "runs", "", "Signed", "today"],
            "conf": ["-1", "90", "80", "70", "-1", "60", 100],
            "block_num": [1, 1, 1, 1, 1, 1, 1], "par_num": [1, 1, 1, 1, 2, 2, 2], "line_num": [0, 1, 1, 2, 0, 1, 1]}
    out = ocr.words_to_text(data)
    assert out.text == "The lease\nruns\n\nSigned today"
    assert out.confidence == pytest.approx((90 + 80 + 70 + 60 + 100) / 5)
    assert ocr.words_to_text({"text": [], "conf": []}) == ocr.OcrText("", 0.0)


@pytest.mark.parametrize("token, kept", [
    # leader-line junk, as Tesseract read real dotted form rows (DocVQA dev pages)
    ("eseseeseeeeeneneeseeeeeeetseeeee", ""), (".cccsees", ""), ("Teeeeeee", ""), ("smeemneennnennmnen", ""),
    ("Excellent....seseees", "Excellent...."), ("services...........cceeeeeee", "services..........."),
    ("Good....eee", "Good....eee"),            # after the dots only 'eee': too short to call junk, and 'Good' must survive
    # real words and values stay, including the ones a looser rule removed on the dev pages
    ("LILLY", "LILLY"), ("Added", "Added"), ("refer", "refer"), ("needed.", "needed."), ("assess", "assess"),
    ("Mississippi", "Mississippi"), ("$10,000", "$10,000"), ("M.D.,", "M.D.,"), ("Indifferent........", "Indifferent........"),
    ("committee", "committee"), ("Ph.D.,", "Ph.D.,"), ("P-09870-00009....", "P-09870-00009...."),
])
def test_leader_line_junk_is_removed_and_real_words_are_kept(token, kept):
    assert ocr.clean_token(token) == kept


def test_cleaning_counts_the_junk_but_the_confidence_still_counts_every_word():
    data = {"text": ["Liquid", "Nitrogen", ".cccsees", "$200"], "conf": [90, 90, 10, 70],
            "block_num": [1] * 4, "par_num": [1] * 4, "line_num": [1] * 4}
    out = ocr.words_to_text(data)
    assert out.text == "Liquid Nitrogen $200" and out.junk == 1
    assert out.confidence == pytest.approx((90 + 90 + 10 + 70) / 4)
    assert ocr.words_to_text(data, clean=False).text == "Liquid Nitrogen .cccsees $200"


def test_a_page_of_nothing_but_junk_leaves_no_text():
    """All words removed leaves no text, so read_image counts the page unreadable instead of indexing noise."""
    data = {"text": ["eseseeseeeee", "cccsseeeee"], "conf": [80, 80], "block_num": [1, 1], "par_num": [1, 1], "line_num": [1, 1]}
    assert ocr.words_to_text(data).text == ""


def test_the_no_text_message_says_why():
    assert "could not be read" in C.no_text_message({"unreadable_pages": [1]})
    assert C.no_text_message({}) == "No readable text was found in this file."


# ------------------------------------------------------------------ end to end: search, cite, access control, delete

def test_a_scanned_document_is_searchable_cited_as_scanned_access_controlled_and_deletable(tmp_path, engine):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("qdrant_client")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from access_policy import DocACL, get_default_policy
    from vector_db import close_qdrant_client
    from encryptor import Encryptor
    from pipeline_config import get_preset
    from rag_pipeline import RAGPipeline

    policy = get_default_policy()
    manager, employee = policy.make_user("mgr", "Manager"), policy.make_user("emp", "Employee")
    cfg = get_preset("dense_only").with_overrides(qdrant_path=str(tmp_path / "q"), collection="t")
    rag = RAGPipeline(cfg, encryptor=Encryptor(key=AESGCM.generate_key(bit_length=256)), policy=policy)
    engine(good(), ocr.OcrText("~~ noise", 12.0))
    try:
        acl = policy.make_acl("Manager", [])
        result = rag.ingest_document(scanned_pdf(tmp_path / "lease.pdf", pages=2), acl)
        assert result["chunks"] == 1 and result["extraction"]["unreadable_pages"] == [2]

        found = rag.retrieve("When was the Rotterdam warehouse lease signed?", manager)
        assert SENTENCE in found.chunks[0]["chunk"]
        assert found.sources() == [{"n": 1, "doc_id": result["doc_id"], "chunk_index": 0, "page": 1, "scanned": True}]
        assert rag.retrieve("When was the Rotterdam warehouse lease signed?", employee).chunks == []   # OCR text keeps the access level

        rag.registry.register(result["doc_id"], acl, "lease.pdf", "mgr", 1234)
        rag.registry.set_status(result["doc_id"], "done", chunks=1, scan=result["extraction"])
        public = rag.registry.get(result["doc_id"], manager).to_public(policy, include_uploader=False)
        assert public["scan"] == {"ocr_pages": 1, "unreadable_pages": [2], "unreadable_images": 0, "skipped": 0, "junk_removed": 0}

        rag.delete_document(result["doc_id"])
        assert rag.store.count(result["doc_id"]) == 0
        assert rag.retrieve("When was the Rotterdam warehouse lease signed?", manager).chunks == []
        assert isinstance(DocACL(min_rank=0), DocACL)
    finally:
        close_qdrant_client(cfg.qdrant_path)


def test_a_document_without_scans_has_no_scan_report_and_a_file_with_no_text_indexes_nothing(tmp_path, engine):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("qdrant_client")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from access_policy import DocACL, get_default_policy
    from vector_db import close_qdrant_client
    from encryptor import Encryptor
    from pipeline_config import get_preset
    from rag_pipeline import RAGPipeline

    policy = get_default_policy()
    cfg = get_preset("dense_only").with_overrides(qdrant_path=str(tmp_path / "q"), collection="t")
    rag = RAGPipeline(cfg, encryptor=Encryptor(key=AESGCM.generate_key(bit_length=256)), policy=policy)
    engine(ocr.OcrText("", 0.0))
    try:
        acl = DocACL(min_rank=0)
        empty = rag.ingest_document(scanned_pdf(tmp_path / "blurry.pdf"), acl)
        assert empty["chunks"] == 0 and empty["extraction"]["unreadable_pages"] == [1]      # nothing indexed, and no crash

        typed = tmp_path / "typed.md"
        typed.write_text("A typed note about the quarterly budget.\n", encoding="utf-8")
        result = rag.ingest_document(str(typed), acl)
        rag.registry.register(result["doc_id"], acl, "typed.md", "emp", 10)
        rag.registry.set_status(result["doc_id"], "done", chunks=result["chunks"], scan=result["extraction"])
        assert "scan" not in rag.registry.get(result["doc_id"]).to_public(policy, include_uploader=False)
    finally:
        close_qdrant_client(cfg.qdrant_path)


# ------------------------------------------------------------------ the real engine (skipped where Tesseract is not installed)

def test_the_real_tesseract_engine_reads_a_clean_printed_line():
    pytest.importorskip("pytesseract")
    try:
        real = ocr.TesseractEngine()
    except Exception:
        pytest.skip("the tesseract program is not installed")
    from PIL import ImageFont
    img = Image.new("RGB", (1400, 200), "white")
    try:
        font = ImageFont.load_default(size=48)
    except TypeError:
        pytest.skip("this Pillow cannot scale its default font")
    ImageDraw.Draw(img).text((30, 60), "The warehouse lease runs for nine years.", fill="black", font=font)
    out = real.read(img)
    assert "warehouse lease" in out.text and out.confidence > 60
