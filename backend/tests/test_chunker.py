"""Chunking: PDF line re-flow, sentence splitting of long paragraphs, page numbers, and unchanged behaviour for plain text."""
import random

import pytest

import chunker as C

EVERYONE_ACL = None


# ------------------------------------------------------------------ a tiny PDF writer (so tests need no extra libraries)

def make_pdf(path, pages):
    """pages: list of pages, each a list of text lines. Writes a minimal valid PDF (Helvetica 11 pt, one line per row)."""
    objs = []

    def add(body):
        objs.append(body)
        return len(objs)

    add("<< /Type /Catalog /Pages 2 0 R >>")                      # object 1
    add("PAGES")                                                    # object 2 (filled in below)
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")  # object 3
    kids = []
    for lines in pages:
        esc = [ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for ln in lines]
        stream = "BT /F1 11 Tf 14 TL 40 760 Td " + " ".join(f"({ln}) Tj T*" for ln in esc) + " ET"
        content = add(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        page = add(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content} 0 R /Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(page)
    objs[1] = f"<< /Type /Pages /Count {len(kids)} /Kids [{' '.join(f'{k} 0 R' for k in kids)}] >>"
    out, offsets = "%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out.encode("latin-1")))
        out += f"{i} 0 obj\n{body}\nendobj\n"
    xref = len(out.encode("latin-1"))
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    with open(path, "wb") as f:
        f.write(out.encode("latin-1"))


def wrap(text, width=60):
    """Hard-wrap text at `width` characters, the way a PDF layout engine does."""
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + ([cur] if cur else [])


PARA1 = ("The quarterly review covered revenue, hiring and the plan for the new warehouse in detail. Revenue grew in every region "
         "and the committee asked for a follow up on the northern figures before the next meeting in March.")
PARA2 = ("Hiring will continue through the spring. Managers should send their requests to the finance team by the first of the month.")


# ------------------------------------------------------------------ re-flow

def test_hard_wrapped_lines_become_real_paragraphs():
    page = "\n".join(wrap(PARA1) + wrap(PARA2))
    paras = C.reflow_pdf_page(page)
    assert paras == [PARA1, PARA2]


def test_a_word_split_by_a_line_end_hyphen_is_rejoined():
    paras = C.reflow_pdf_page("The new informa-\ntion system will be rolled out to every office in\nthe region by the end of the year.")
    assert paras == ["The new information system will be rolled out to every office in the region by the end of the year."]


def test_list_items_stay_separate_and_blank_lines_split_paragraphs():
    text = "Agenda for the meeting of the whole team on Monday morning at nine\n- budget review for the next quarter ahead\n- hiring plan for the spring\n\nNotes follow below."
    paras = C.reflow_pdf_page(text)
    assert len(paras) == 4 and paras[1].startswith("- budget") and paras[2].startswith("- hiring") and paras[3] == "Notes follow below."


# ------------------------------------------------------------------ long paragraphs

def sentence(i, words=20):
    return " ".join(f"w{i}x{j}" for j in range(words - 1)) + f" end{i}."


def test_a_long_paragraph_is_split_at_sentence_boundaries(tmp_path):
    f = tmp_path / "long.txt"
    f.write_text(" ".join(sentence(i) for i in range(20)) + "\n", encoding="utf-8")            # 400 words in ONE line
    split = C.run_chunking(str(f), chunk_size=100, overlap=0)
    whole = C.run_chunking(str(f), chunk_size=100, overlap=0, split_long=False)
    assert len(whole) == 1 and len(whole[0].split()) == 400                                      # the old, oversized chunk
    assert len(split) >= 4 and all(len(c.split()) <= 100 for c in split)
    assert all(c.rstrip().endswith(".") for c in split)                                         # never cut mid-sentence
    assert " ".join(split).split() == whole[0].split()                                          # nothing lost or reordered


def test_one_enormous_sentence_is_cut_by_words(tmp_path):
    f = tmp_path / "run.txt"
    f.write_text(" ".join(f"word{i}" for i in range(350)) + "\n", encoding="utf-8")
    chunks = C.run_chunking(str(f), chunk_size=100, overlap=0)
    assert [len(c.split()) for c in chunks] == [100, 100, 100, 50]


# ------------------------------------------------------------------ pages

def test_chunks_remember_their_pdf_pages_and_pages_are_never_glued(tmp_path):
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf, [wrap(PARA1), wrap(PARA2), wrap("Final page text about the northern region and the March follow up meeting agenda.")])
    chunks = C.run_chunking_with_pages(str(pdf), chunk_size=25, overlap=0)
    assert [c.page_start for c in chunks][0] == 1 and chunks[-1].page_end == 3
    assert all(c.page_start <= c.page_end for c in chunks)
    text = " ".join(c.text for c in chunks)
    assert "March.Hiring" not in text and "March. Hiring" in text.replace("\n", " ")             # a page break does not glue words
    assert any(c.page_start != c.page_end for c in chunks) or len(chunks) >= 3


def test_pdf_paragraphs_are_not_one_chunk_per_visual_line(tmp_path):
    pdf = tmp_path / "wrapped.pdf"
    make_pdf(pdf, [wrap(PARA1) + wrap(PARA2)])
    chunks = C.run_chunking(str(pdf), chunk_size=150, overlap=0)
    assert len(chunks) == 1 and PARA1 in chunks[0] and PARA2 in chunks[0]                       # previously: one "paragraph" per line


def test_plain_text_has_no_pages(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("First line here.\nSecond line here.\n", encoding="utf-8")
    assert all(c.page_start is None and c.page_end is None for c in C.run_chunking_with_pages(str(f)))


# ------------------------------------------------------------------ nothing changes for line-per-passage corpora

def old_run_chunking_text(text, chunk_size=150, overlap=20):
    """The original chunker's algorithm, kept here as the reference: the benchmark corpora must chunk exactly like this."""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks, current, current_length = [], [], 0
    for para in paragraphs:
        para_len = len(para.split())
        if current_length + para_len > chunk_size and current:
            chunks.append("\n".join(current))
            overlap_chunk, overlap_length = [], 0
            for prev_para in reversed(current):
                if overlap_length + len(prev_para.split()) <= overlap:
                    overlap_chunk.insert(0, prev_para)
                    overlap_length += len(prev_para.split())
                else:
                    break
            current, current_length = overlap_chunk, overlap_length
        current.append(para)
        current_length += para_len
    if current:
        chunks.append("\n".join(current))
    return chunks


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_without_sentence_splitting_the_chunks_equal_the_original_algorithm(tmp_path, seed):
    rng = random.Random(seed)
    lines = [" ".join(f"t{rng.randrange(500)}" for _ in range(rng.choice([3, 12, 40, 90, 160, 380]))) + "." for _ in range(60)]
    f = tmp_path / "corpus.md"
    f.write_text("# Title\n\n" + "\n\n".join(lines) + "\n", encoding="utf-8")
    assert C.run_chunking(str(f), 150, 20, split_long=False) == old_run_chunking_text(f.read_text(encoding="utf-8"))


def test_text_extraction_never_glues_pages(tmp_path):
    pdf = tmp_path / "two.pdf"
    make_pdf(pdf, [["Page one ends here."], ["Page two starts here."]])
    assert "here.Page" not in C.extract_text_from_file(str(pdf))
