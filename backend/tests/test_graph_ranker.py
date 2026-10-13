"""Hop-aware ranker and the passage-title helper (no database needed)."""
import graph_ranker as GR
from kg_builder import chunk_title_names, entity_key


def cand(**kw):
    base = {f: 0.0 for f in GR.FEATURES}
    base.update(kw)
    return base


def test_a_page_of_a_question_entity_can_outrank_a_more_topical_passage():
    topical = cand(ce=8.0)
    page = cand(ce=1.0, page_q=1, cover=1, cover_frac=0.5)
    assert GR.score(page) > GR.score(topical)


def test_without_graph_signals_the_order_is_the_cross_encoder_order():
    a, b = cand(ce=3.0), cand(ce=1.0)
    assert GR.score(a) > GR.score(b)


def test_custom_weights_are_used_and_unknown_features_ignored():
    assert GR.score(cand(ce=2.0), {"ce": 1.0, "nonsense": 5.0}) == 2.0


def test_feature_row_follows_the_feature_order():
    row = GR.feature_row(cand(ce=2.0, page_q=1))
    assert row[GR.FEATURES.index("ce")] == 2.0 and row[GR.FEATURES.index("page_q")] == 1.0


def test_titles_of_encyclopedia_style_passages_are_found():
    assert chunk_title_names("Kool Kojak. Allan P. Grigg, better known as Kool Kojak, is an American musician.") == ["Kool Kojak"]
    assert chunk_title_names("Fort Lee, New Jersey. Fort Lee is a borough in Bergen County.") == ["Fort Lee, New Jersey", "Fort Lee"]
    assert chunk_title_names("The Lorax (film). The Lorax is a 2012 film.") == ["The Lorax"]


def test_ordinary_sentences_are_not_mistaken_for_titles():
    assert chunk_title_names("The committee met on Tuesday to review the quarterly budget and approve the plan. Next steps follow.") == []
    assert chunk_title_names("No full stop here at all") == []
    assert chunk_title_names("A. B. Smith joined.") == []


def test_title_keys_match_how_the_question_entities_are_looked_up():
    assert entity_key(chunk_title_names("The Lorax (film). The Lorax is a film.")[0]) == "lorax"
