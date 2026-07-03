# Tests for the reflection node's citation checker (app/agents/synthesis_graph.py).
#
# These only exercise the pure Python logic — extract_citations() and
# validate_citations() never call the LLM or touch a database, so these
# tests run in milliseconds with no external services and no API keys.
# That's deliberately what we're testing first: the safety net that
# catches a hallucinated citation before it reaches a user, since that's
# the whole point of the reflect node in the LangGraph pipeline.

from app.agents.synthesis_graph import extract_citations, validate_citations

PAPERS = [
    {"title": "Attention Is All You Need"},
    {"title": "Deep Residual Learning for Image Recognition"},
]


def test_extract_citations_finds_every_bracketed_title():
    text = (
        "This builds on [Attention Is All You Need] and also cites "
        "[Deep Residual Learning for Image Recognition]."
    )
    assert extract_citations(text) == [
        "Attention Is All You Need",
        "Deep Residual Learning for Image Recognition",
    ]


def test_extract_citations_returns_empty_list_when_none_present():
    assert extract_citations("No citations in this sentence at all.") == []


def test_validate_citations_accepts_exact_title_match():
    valid, invalid = validate_citations(["Attention Is All You Need"], PAPERS)
    assert valid == ["Attention Is All You Need"]
    assert invalid == []


def test_validate_citations_accepts_positional_paper_n():
    valid, invalid = validate_citations(["Paper 2"], PAPERS)
    assert valid == ["Paper 2"]
    assert invalid == []


def test_validate_citations_accepts_close_fuzzy_match():
    # Same title, different casing — should still match via difflib.
    valid, invalid = validate_citations(["Attention is all you need"], PAPERS)
    assert valid == ["Attention is all you need"]
    assert invalid == []


def test_validate_citations_flags_hallucinated_title():
    valid, invalid = validate_citations(["A Completely Made Up Paper"], PAPERS)
    assert valid == []
    assert invalid == ["A Completely Made Up Paper"]


def test_validate_citations_flags_out_of_range_paper_n():
    valid, invalid = validate_citations(["Paper 9"], PAPERS)
    assert valid == []
    assert invalid == ["Paper 9"]
