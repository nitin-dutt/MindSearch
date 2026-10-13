"""Hop-aware ranking of candidate passages (alignment_plan: GraphRAG multi-hop).

The search pool plus the passages the graph links in are scored by a small linear model over a few features. The model is
trained offline (evaluation/hop_tune.py: logistic regression, cross-validated by question) and only its weights live here, so
nothing but arithmetic runs at query time.

Features of one candidate passage:
    ce          cross-encoder score of (question, passage)
    ce_gap      ce minus the best ce among the candidates (0 for the best)
    cover       how many of the question's entities the passage mentions (from the graph)
    cover_frac  cover divided by the number of question entities
    cover_s     how many entities of the top passages (the bridge) it mentions, capped at 5
    page_q      1 if it is the page (title) of an entity named in the question
    page_s      1 if it is the page of an entity found in the top passages
    via_graph   1 if only the graph, not the search, proposed it
    pool_rank   1 / (1 + position in the hybrid search ranking); 0 if the search did not return it
"""
FEATURES = ["ce", "ce_gap", "cover", "cover_frac", "cover_s", "page_q", "page_s", "via_graph", "pool_rank"]

# Weights fitted by evaluation/hop_tune.py on the 200 HotpotQA questions (logistic regression on: ce, cover, cover_frac, page_q,
# page_s; the other features did not help in cross-validation, so their weight is 0). Only the ORDER of scores matters.
DEFAULT_WEIGHTS = {"ce": 0.31, "ce_gap": 0.0, "cover": -0.26, "cover_frac": 1.98, "cover_s": 0.0, "page_q": 3.28, "page_s": 2.99,
                   "via_graph": 0.0, "pool_rank": 0.0}


def feature_row(c) -> list:
    return [float(c.get(f, 0.0)) for f in FEATURES]


def score(c, weights=None) -> float:
    w = weights or DEFAULT_WEIGHTS
    return sum(w.get(f, 0.0) * float(c.get(f, 0.0)) for f in FEATURES)
