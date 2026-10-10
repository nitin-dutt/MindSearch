from evaluation import calibration as C

FILLED = """# sheet
## 1

**Question:** q1

- Correct (Y/N): y
- Supported (Y/N): N

---

## 2

- Correct (Y/N): N
- Supported (Y/N):

---
## 3
- Correct (Y/N): Y
- Supported (Y/N): Y
"""


def test_parse_sheet_reads_labels_and_flags_blanks():
    got = C.parse_sheet(FILLED)
    assert got[1] == {"correct": True, "supported": False}
    assert got[2] == {"correct": False, "supported": None}   # blank line = unfilled
    assert got[3] == {"correct": True, "supported": True}


def test_agreement_counts_disagreements_in_both_directions():
    pairs = [(True, True), (False, False), (False, True), (True, False), (True, True)]
    a = C.agreement(pairs)
    assert a["n"] == 5 and a["agreement"] == 3 / 5
    assert a["judge_ok_human_bad"] == 1 and a["judge_bad_human_ok"] == 1


def test_pick_items_always_includes_clearly_low_scores_and_hides_order():
    rows = [{"id": str(i), "faithfulness": f} for i, f in
            enumerate([0.0, 0.3] + [0.7] * 30 + [1.0] * 40)]
    chosen = C.pick_items(rows, n=20)
    assert len(chosen) == 20
    assert {"0", "1"} <= {r["id"] for r in chosen}           # both low items present
    assert sum(1 for r in chosen if r["faithfulness"] >= 0.999) >= 5
