"""Serving-path tests: the forecast contract, and the bugs that were already found once.

Each of these corresponds to something that was actually broken, not to something
imagined. A test that has only ever passed is not a check.

Run:  .venv/bin/python -m pytest -q tests/test_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "scripts"))

DB = ROOT / "data" / "faffabout.duckdb"
MODELS = ROOT / "baseline" / "models"
SEARCH = ROOT / "data" / "search" / "archiveDB.idx"

needs_archive = pytest.mark.skipif(
    not (DB.exists() and (MODELS / "gate_0.txt").exists()),
    reason="needs the built archive and the trained boosters")
needs_search = pytest.mark.skipif(not SEARCH.exists(), reason="needs the MMseqs2 index")

LYSOZYME = ("MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILR"
            "NAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL")


# --------------------------------------------------------------------------- choices

def test_cold_shock_is_derived_from_the_host():
    """`host` collapses to five archive values, so BL21 and Arctic Express are the same
    value and produced byte-identical forecasts until cold_shock was mapped across."""
    import predict as P
    assert P.Choices(host="arctic").cold_shock is True
    assert P.Choices(host="bl21").cold_shock is False
    assert P.Choices().cold_shock is False


def test_declaring_any_protocol_field_counts_as_declared():
    import predict as P
    assert P.Choices().declares_protocol is False
    for f in ("host", "tag", "protease"):
        assert P.Choices(**{f: "his" if f == "tag" else "tev" if f == "protease" else "bl21"}).declares_protocol


def test_cold_shock_reaches_the_feature_row():
    import predict as P
    row = P.feature_row(LYSOZYME, "Escherichia coli", "83333", {"per_gate": {}},
                        P.Choices(host="arctic", tag="his"), 0)
    assert row["cold_shock"] is True
    assert row["host"] == "ecoli"
    row2 = P.feature_row(LYSOZYME, "Escherichia coli", "83333", {"per_gate": {}},
                         P.Choices(host="bl21", tag="his"), 0)
    assert row2["cold_shock"] is False
    assert row2["host"] == "ecoli", "both are E. coli; only cold_shock may separate them"


# --------------------------------------------------------------------------- arithmetic

def test_survival_is_the_running_product_and_starts_at_one():
    import predict as P
    s = P.survival([0.9, 0.5, 0.8])
    assert s[0] == 1.0
    assert s[1] == pytest.approx(0.9)
    assert s[3] == pytest.approx(0.9 * 0.5 * 0.8)
    assert all(s[i] >= s[i + 1] for i in range(len(s) - 1)), "survival can only fall"


def test_bottleneck_is_the_largest_absolute_drop():
    import predict as P
    surv = P.survival([0.95, 0.90, 0.20, 0.99])
    assert P.bottleneck(surv)["gate"] == 2


def test_jeffreys_interval_widens_as_evidence_thins():
    import predict as P
    wide = P.jeffreys_interval(2, 3)
    narrow = P.jeffreys_interval(200, 300)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])
    assert P.jeffreys_interval(0, 0) == (0.0, 1.0), "no evidence means no constraint"


# --------------------------------------------------------------------------- taxonomy

@needs_archive
def test_taxonomy_table_resolves_by_id_and_by_name():
    import taxo
    if not taxo.available():
        pytest.skip("taxonomy_lookup not built")
    assert taxo.classify("83333", "")["genus"] == "Escherichia"
    assert taxo.classify("", "Homo sapiens")["superkingdom"] == "Eukaryota"


@needs_archive
def test_strain_suffixes_fall_back_to_the_species():
    """"Arabidopsis thaliana Columbia" has no exact key; the prefix ladder must find it."""
    import taxo
    if not taxo.available():
        pytest.skip("taxonomy_lookup not built")
    assert taxo.classify("", "Arabidopsis thaliana Columbia")["superkingdom"] == "Eukaryota"
    assert taxo.classify("", "Pyrococcus furiosus DSM 3638")["superkingdom"] == "Archaea"


@needs_archive
def test_unknown_organism_is_empty_not_an_exception():
    import taxo
    assert taxo.classify("", "Nonexistent blahblah organism")["superkingdom"] == ""


# --------------------------------------------------------------------------- retrieval

@needs_search
def test_retrieval_respects_the_clustering_thresholds():
    import retrieve as RET
    assert RET.MIN_IDENTITY == 0.30, "must match scripts/04_cluster_sequences.sh"
    assert RET.MIN_COVERAGE == 0.80
    assert RET.CLOSE_IDENTITY == 0.70


@needs_search
@needs_archive
def test_censored_precedents_are_retained_and_flagged():
    """Spec rule 1: censored is excluded from the loss but present in context."""
    import duckdb
    import retrieve as RET
    con = duckdb.connect(str(DB), read_only=True)
    precs = RET.precedents(LYSOZYME, con=con)
    ctx = RET.context(precs)
    if not precs:
        pytest.skip("no precedent for this query")
    assert ctx["n_precedents"] <= len(precs), "rates exclude censored, the list does not"
    for p in precs:
        if p.censored:
            assert "censored" in p.note


@needs_search
@needs_archive
def test_context_base_rates_ignore_censored_records():
    import duckdb
    import retrieve as RET
    con = duckdb.connect(str(DB), read_only=True)
    precs = RET.precedents(LYSOZYME, con=con)
    if not any(p.censored for p in precs):
        pytest.skip("no censored precedent in this neighbourhood")
    ctx = RET.context(precs)
    assert ctx["n_precedents"] == sum(1 for p in precs if not p.censored and p.max_stage is not None)
    assert ctx["n_censored"] == sum(1 for p in precs if p.censored)


# --------------------------------------------------------------------------- contract

@needs_search
@needs_archive
def test_predict_returns_the_documented_contract():
    from server import app
    c = app.test_client()
    r = c.post("/api/predict", json={"query": LYSOZYME, "centre": "JCSG",
                                     "host": "bl21", "tag": "his"})
    assert r.status_code == 200
    d = r.json
    for key in ("target", "features", "ladder", "survival", "conditional", "bottleneck",
                "evidence", "precedents", "counterfactuals", "caveats", "model"):
        assert key in d, key
    assert len(d["ladder"]) == 9
    assert len(d["conditional"]) == 8
    assert len(d["survival"]) == 9
    assert 0 <= d["bottleneck"]["gate"] <= 7


@needs_search
@needs_archive
def test_the_model_never_supplies_a_number():
    """Spec section 7.4: the narrative is the only field a language model writes."""
    from server import app
    d = app.test_client().post("/api/predict", json={"query": LYSOZYME}).json
    assert set(d["model"]) == {"probabilities_from", "narrative_from", "narrative"}
    assert d["model"]["probabilities_from"].startswith("lightgbm")


@needs_search
@needs_archive
def test_two_hosts_that_differ_only_in_temperature_give_different_forecasts():
    """The regression this file exists for."""
    from server import app
    c = app.test_client()
    a = c.post("/api/predict", json={"query": LYSOZYME, "host": "bl21", "tag": "his"}).json
    b = c.post("/api/predict", json={"query": LYSOZYME, "host": "arctic", "tag": "his"}).json
    assert a["conditional"] != b["conditional"], \
        "BL21 and Arctic Express collapse to the same archive host; cold_shock must separate them"


@needs_search
@needs_archive
def test_no_counterfactuals_are_offered_without_a_declared_protocol():
    from server import app
    d = app.test_client().post("/api/predict", json={"query": LYSOZYME}).json
    assert d["counterfactuals"] == []
    assert any("counterfactual" in c.lower() for c in d["caveats"])


def test_junk_input_is_a_400_with_an_explanation():
    from server import app
    r = app.test_client().post("/api/predict", json={"query": "not a protein at all"})
    assert r.status_code == 400
    assert "could not tell" in r.json["error"]


@needs_archive
def test_healthz_reports_each_piece_separately():
    from server import app
    d = app.test_client().get("/healthz").json
    for k in ("duckdb", "search_index", "boosters_headline", "boosters_declared", "taxonomy"):
        assert k in d["checks"]


@needs_archive
def test_the_page_renders_with_no_unrendered_template_tags():
    import re
    from server import app
    html = app.test_client().get("/").data.decode()
    assert re.search(r"\{\{|\{%", html) is None
    assert "Attrition pipeline" in html
