"""OCR accuracy and speed on real scanned pages (alignment_plan RET-06). Judge-free.

    python -m evaluation.ocr_eval fetch                               # downloads the pages of the frozen test set
    python -m evaluation.ocr_eval run                                 # before/after OCR on the frozen test set
    python -m evaluation.ocr_eval select --name ocr_docvqa_dev20 --start 600 --exclude ocr_docvqa20   # a DEV set (once)
    python -m evaluation.ocr_eval fetch --set ocr_docvqa_dev20
    python -m evaluation.ocr_eval junk  --set ocr_docvqa_dev20        # what the leader-junk cleaner removes, and answers kept

Sets (`--set`, a manifest eval_data/<name>.json plus the pages in eval_data/<name>/):
  ocr_docvqa20       FROZEN test set: 20 scanned pages, 54 questions. Report on it; never choose a setting by looking at it.
  ocr_docvqa_dev20   dev set chosen by the same rules from later rows and other documents: settings are chosen here.
Pages come from DocVQA (real documents from the UCSF Industry Documents Library) with human-written answers. The images are not
in the repository (eval_data/ is git-ignored, and the original DocVQA terms are research use).

Each page is tested in the two ways a user brings a scan in:
  jpg   the page uploaded as a picture
  pdf   the page inside an image-only PDF (what a scanner produces)
through the real code, so the confidence filters, size limits and page rendering are measured too.

Metric: answer found = one of the human answers appears, as whole words (case, spacing and punctuation ignored), in
  indexed   the text of the passages the chunker produces, i.e. what is actually stored and searched (the headline number:
            it includes re-joining words split by a line-end hyphen), and
  raw       the text OCR returned, before chunking.
This measures whether the fact became searchable text; whether the chat then finds and states it is the end-to-end test,
done by uploading the same files through the app. 'before' runs with OCR switched off and is expected to be 0.
Run it inside the worker container (it needs the tesseract program); CPU and Tesseract version are recorded.
"""
import argparse
import json
import os
import platform
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from evaluation.datasets import EVAL_DATA_DIR
from evaluation.generate import RESULTS_DIR
from evaluation.metrics import bootstrap_ci

FROZEN = "ocr_docvqa20"
ROWS_API = ("https://datasets-server.huggingface.co/rows?dataset=lmms-lab-encoder/DocVQA&config=DocVQA"
            "&split=validation&offset={offset}&length={length}")
HEADERS = {"User-Agent": "mindsearch-eval"}

# Selection rules shared by every set (the frozen set was chosen with exactly these, see its manifest)
SKIP_TYPES = {"handwritten", "figure/diagram", "Image/Photo", "Yes/No"}
GENERIC = re.compile(r"(name of the company\?|page no|^what is (the )?date\??$|last (rectangle|picture)|title of the document"
                     r"|year mentioned at the top)", re.I)


def manifest_path(name):
    return os.path.join(EVAL_DATA_DIR, f"{name}.json")


def load_manifest(name):
    path = manifest_path(name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def page_files(name, page):
    base = os.path.join(EVAL_DATA_DIR, name, page["id"])
    return {"jpg": base + ".jpg", "pdf": base + ".pdf"}


def normalise(text):
    """Lower case, every run of punctuation and spaces becomes one space: '$975.00' -> '975 00', 'A.C. Monk' -> 'a c monk'."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.lower()).split())


def answer_found(text, answers):
    """Whole-word match, so a short answer such as '2' is not 'found' inside '2012'."""
    flat = f" {normalise(text)} "
    return any(normalise(a) and f" {normalise(a)} " in flat for a in answers)


def _get_json(url, attempts=5):
    """The rows API answers 502 now and then under load; wait and try again a few times."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == attempts - 1:
                raise
            time.sleep(3 * (attempt + 1))


# ------------------------------------------------------------------ choosing and downloading pages

def select(args):
    """Choose a new set with the frozen set's rules from rows [start, start+rows), skipping every document of the --exclude sets."""
    excluded = set()
    for name in args.exclude:
        excluded |= {p["ucsf_document_id"] for p in load_manifest(name)["pages"]}
    pages, order = {}, []
    for offset in range(args.start, args.start + args.rows, 100):
        for item in _get_json(ROWS_API.format(offset=offset, length=100))["rows"]:
            r = item["row"]
            key = f"{r['ucsf_document_id']}_{r['ucsf_document_page_no']}"
            if key not in pages:
                pages[key] = {"id": key, "ucsf_document_id": r["ucsf_document_id"], "ucsf_page": r["ucsf_document_page_no"],
                              "row_idx": item["row_idx"], "width": r["image"]["width"], "height": r["image"]["height"],
                              "questions": []}
                order.append(key)
            question = " ".join(r["question"].split())
            if (not SKIP_TYPES & set(r["question_types"]) and not any(a.strip().lower() in ("yes", "no") for a in r["answers"])
                    and not GENERIC.search(question)):
                pages[key]["questions"].append({"qid": r["questionId"], "row_idx": item["row_idx"], "question": question,
                                                "types": r["question_types"], "answers": r["answers"]})
    chosen, seen = [], set(excluded)
    for key in order:
        p = pages[key]
        if len(p["questions"]) < 2 or p["width"] < 1000 or p["ucsf_document_id"] in seen:
            continue
        seen.add(p["ucsf_document_id"])
        p["row_idx"] = p["questions"][0]["row_idx"]
        p["questions"] = p["questions"][:3]
        chosen.append(p)
        if len(chosen) == args.n:
            break
    manifest = {"name": args.name, "role": "dev: choose settings here",
                "source": "DocVQA (Mathew et al., 2021, arXiv:2007.00398), validation split, via the Hugging Face mirror "
                          "lmms-lab-encoder/DocVQA, config DocVQA",
                "page_images_from": "UCSF Industry Documents Library (scanned real documents)",
                "selection": f"validation rows {args.start}-{args.start + args.rows - 1} in order, same rules as {FROZEN}; documents of "
                             f"{', '.join(args.exclude) or 'no other set'} excluded",
                "license_note": "Images are fetched locally and never committed (original DocVQA terms are research use).",
                "pages": chosen}
    with open(manifest_path(args.name), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote {manifest_path(args.name)}: {len(chosen)} pages, {sum(len(p['questions']) for p in chosen)} questions")


def fetch(args):
    from PIL import Image
    manifest = load_manifest(args.set)
    os.makedirs(os.path.join(EVAL_DATA_DIR, args.set), exist_ok=True)
    for page in manifest["pages"]:
        files = page_files(args.set, page)
        if not os.path.exists(files["jpg"]):
            # The image link in the API answer is signed and expires, so ask for a fresh one for this exact row.
            row = _get_json(ROWS_API.format(offset=page["row_idx"], length=1))["rows"][0]["row"]
            if row["ucsf_document_id"] != page["ucsf_document_id"] or str(row["ucsf_document_page_no"]) != str(page["ucsf_page"]):
                raise SystemExit(f"Row {page['row_idx']} is not page {page['id']} any more: the mirror changed, stop.")
            with urllib.request.urlopen(urllib.request.Request(row["image"]["src"], headers=HEADERS), timeout=120) as r:
                data = r.read()
            with open(files["jpg"], "wb") as f:
                f.write(data)
        if not os.path.exists(files["pdf"]):
            with Image.open(files["jpg"]) as im:
                im.convert("RGB").save(files["pdf"], "PDF", resolution=200.0)   # image only: no text layer, like a scanner
        print(f"  {page['id']}: {os.path.getsize(files['jpg']) // 1024} KB")
    print(f"Pages are in {os.path.join(EVAL_DATA_DIR, args.set)}")


def require_pages(name, manifest):
    missing = [p["id"] for p in manifest["pages"] if not all(os.path.exists(f) for f in page_files(name, p).values())]
    if missing:
        raise SystemExit(f"{len(missing)} page files missing (e.g. {missing[0]}). Run: python -m evaluation.ocr_eval fetch --set {name}")


# ------------------------------------------------------------------ measuring

def extract(path):
    """(raw OCR text, indexed passage text, extraction report, seconds), through the same code an upload uses."""
    from chunker import chunk_document
    started = time.perf_counter()
    chunks, report = chunk_document(path)
    seconds = time.perf_counter() - started
    from chunker import extract_document
    pages, _ = extract_document(path)                 # the raw text, for the 'raw' column only (not timed)
    return "\n".join(p.text for p in pages), "\n".join(c.text for c in chunks), report.to_dict(), seconds


def tesseract_version():
    try:
        import pytesseract
        return str(pytesseract.get_tesseract_version())
    except Exception as e:  # noqa: BLE001
        return f"not available ({type(e).__name__})"


def run(args):
    import ocr
    from settings import get_settings
    manifest = load_manifest(args.set)
    require_pages(args.set, manifest)

    cfg = get_settings()
    rows, page_rows = [], []
    for stage in ("before", "after"):
        if stage == "before":
            ocr.set_engine(None)            # OCR off: what the system did before RET-06
        else:
            ocr.reset_engine()
            if ocr.get_engine() is None:
                raise SystemExit("Tesseract is not available here. Run this inside the worker container.")
        for page in manifest["pages"]:
            for fmt in args.formats:
                raw, indexed, report, seconds = extract(page_files(args.set, page)[fmt])
                found = [answer_found(indexed, q["answers"]) for q in page["questions"]]
                found_raw = [answer_found(raw, q["answers"]) for q in page["questions"]]
                page_rows.append({"stage": stage, "format": fmt, "page": page["id"], "seconds": seconds, "chars": len(indexed),
                                  "report": report, "found": sum(found), "found_raw": sum(found_raw), "questions": len(found)})
                for q, ok, ok_raw in zip(page["questions"], found, found_raw):
                    rows.append({"stage": stage, "format": fmt, "page": page["id"], "qid": q["qid"], "question": q["question"],
                                 "answers": q["answers"], "found": ok, "found_raw": ok_raw})
                print(f"  {stage:<6} {fmt} {page['id']:<14} {seconds:6.2f}s  {len(indexed):5d} chars  answers found {sum(found)}/{len(found)}"
                      f" (raw {sum(found_raw)})  junk removed {report.get('junk_removed', 0)}")
    ocr.reset_engine()

    summary = {"set": args.set, "frozen": args.set == FROZEN, "n_pages": len(manifest["pages"]),
               "n_questions": sum(len(p["questions"]) for p in manifest["pages"]),
               "tesseract": tesseract_version(), "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
               "settings": {k: getattr(cfg, k) for k in ("ocr_languages", "ocr_dpi", "ocr_min_confidence",
                                                         "ocr_min_chars", "ocr_page_timeout_s")},
               "results": {}}
    for stage in ("before", "after"):
        for fmt in args.formats:
            qs = [r for r in rows if r["stage"] == stage and r["format"] == fmt]
            ps = [r for r in page_rows if r["stage"] == stage and r["format"] == fmt]
            secs = sorted(p["seconds"] for p in ps)
            entry = {}
            for key, label in (("found", "answer_found"), ("found_raw", "answer_found_raw")):
                mean, lo, hi, n = bootstrap_ci([float(r[key]) for r in qs])
                entry[label] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": n, "count": sum(r[key] for r in qs)}
            entry.update({
                "pages_with_text": sum(p["chars"] > 0 for p in ps),
                "pages_reported_unreadable": sum(bool(p["report"]["unreadable_pages"] or p["report"]["unreadable_images"]) for p in ps),
                "junk_removed": sum(p["report"].get("junk_removed", 0) for p in ps),
                "seconds_per_page_mean": sum(secs) / len(secs), "seconds_per_page_median": secs[len(secs) // 2],
                "seconds_per_page_max": secs[-1]})
            summary["results"][f"{stage}/{fmt}"] = entry

    run_dir = os.path.join(RESULTS_DIR, f"ocr__{args.set}__{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    for name, data in (("questions.jsonl", rows), ("pages.jsonl", page_rows)):
        with open(os.path.join(run_dir, name), "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n=== OCR on {args.set}: {summary['n_pages']} scanned pages, {summary['n_questions']} questions, Tesseract {summary['tesseract']} ===")
    for key, r in summary["results"].items():
        a, b = r["answer_found"], r["answer_found_raw"]
        print(f"  {key:<11} answer in indexed text {a['mean']:.1%} [{a['ci_low']:.1%}, {a['ci_high']:.1%}] ({a['count']}/{a['n']})"
              f"  raw {b['mean']:.1%}  pages with text {r['pages_with_text']}/{summary['n_pages']}  unreadable "
              f"{r['pages_reported_unreadable']}  junk removed {r['junk_removed']}  s/page mean {r['seconds_per_page_mean']:.2f}"
              f" max {r['seconds_per_page_max']:.2f}")
    print(f"\nSaved {run_dir}")


def junk(args):
    """The leader-line cleaner (ocr.clean_token) on vs off: answers found in the indexed text, and every token it removed or
    trimmed, so a person can check each one is junk and not real text. Tesseract runs once per page. The rule was chosen by
    looking at the DEV set only; on the frozen set this command only reports what the fixed rule does."""
    import ocr
    from chunker import reflow_pdf_page
    manifest = load_manifest(args.set)
    require_pages(args.set, manifest)
    engine = ocr.get_engine()
    if engine is None:
        raise SystemExit("Tesseract is not available here. Run this inside the worker container.")
    totals = {"off": 0, "on": 0, "questions": 0, "tokens": 0, "changed": 0}
    for page in manifest["pages"]:
        with open(page_files(args.set, page)["jpg"], "rb") as f:
            data = engine.word_table(ocr.open_image(f.read(), 25_000_000))
        for mode, clean in (("off", False), ("on", True)):
            text = "\n".join(reflow_pdf_page(ocr.words_to_text(data, clean=clean).text))
            totals[mode] += sum(answer_found(text, q["answers"]) for q in page["questions"])
        totals["questions"] += len(page["questions"])
        for word in data["text"]:
            word = (word or "").strip()
            if not word:
                continue
            totals["tokens"] += 1
            kept = ocr.clean_token(word)
            if kept != word:
                totals["changed"] += 1
                print(f"  {page['id']:<14} {word[:45]!r:>48} -> {kept!r}")
    print(f"\n{args.set}: answers in indexed text {totals['off']}/{totals['questions']} without the cleaner, {totals['on']}/"
          f"{totals['questions']} with it; {totals['changed']} of {totals['tokens']} tokens removed or trimmed")
    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sel = sub.add_parser("select")
    sel.add_argument("--name", required=True)
    sel.add_argument("--start", type=int, required=True)
    sel.add_argument("--rows", type=int, default=600)
    sel.add_argument("--n", type=int, default=20)
    sel.add_argument("--exclude", nargs="*", default=[FROZEN])
    fe = sub.add_parser("fetch")
    fe.add_argument("--set", default=FROZEN)
    rn = sub.add_parser("run")
    rn.add_argument("--set", default=FROZEN)
    rn.add_argument("--formats", nargs="+", choices=("jpg", "pdf"), default=["jpg", "pdf"])
    ju = sub.add_parser("junk")
    ju.add_argument("--set", required=True)
    args = ap.parse_args()
    {"select": select, "fetch": fetch, "run": run, "junk": junk}[args.cmd](args)


if __name__ == "__main__":
    main()
