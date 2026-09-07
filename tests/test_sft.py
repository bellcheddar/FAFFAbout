"""SFT corpus tests: the negative-mining rules and the no-hallucination discipline.

Run:  .venv/bin/python -m pytest -q tests/test_sft.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

sft = __import__("07_build_sft")

TRAIN = ROOT / "data" / "sft" / "train.jsonl"
TASKS = ROOT / "data" / "sft" / "train_tasks.jsonl"
needs_corpus = pytest.mark.skipif(not TRAIN.exists(), reason="needs the built corpus")

PRIORS = {g: p for g, p in enumerate([0.77, 0.64, 0.69, 0.76, 0.37, 0.76, 0.75, 0.97])}


def row(**kw):
    base = dict(target_id="JCSG-1", centre="JCSG", gate=0, label="cleared", max_stage=4,
                organism="Escherichia coli", superkingdom="Bacteria", seq_len=300,
                pi=6.5, gravy=-0.2, net_charge_ph7=-3.0, cys_count=2, tm_helices=0,
                signal_peptide=False, disorder_frac=0.1, disorder_nterm=0.1,
                disorder_cterm=0.1, low_complexity_frac=0.02, protein_types="",
                target_construct_type="", host="ecoli", tag="his", protease="tev",
                construct_start=None, construct_end=None, n_precedents=10,
                n_precedents_cleared=8, cluster_base_rate=0.8, n_close_precedents=2,
                cluster_censored_frac=0.05, precedents=[], gate_rates={})
    base.update(kw)
    return base


# --------------------------------------------------------------------------- calibration

def test_probability_does_not_depend_on_this_targets_own_outcome():
    """Training 0.85 for every success and 0.15 for every failure teaches confident
    guessing, which is exactly the overconfidence the spec warns about."""
    a = sft.calibrated_p(row(label="cleared"), PRIORS)
    b = sft.calibrated_p(row(label="failed"), PRIORS)
    assert a == b


def test_probability_follows_the_cluster_evidence():
    good = sft.calibrated_p(row(n_precedents=20, n_precedents_cleared=19), PRIORS)
    bad = sft.calibrated_p(row(n_precedents=20, n_precedents_cleared=2), PRIORS)
    assert good > 0.8 and bad < 0.3


def test_thin_evidence_is_shrunk_towards_the_global_gate_rate():
    """One precedent clearing is not evidence of a 100% success rate."""
    thin = sft.calibrated_p(row(gate=4, n_precedents=1, n_precedents_cleared=1), PRIORS)
    thick = sft.calibrated_p(row(gate=4, n_precedents=200, n_precedents_cleared=200), PRIORS)
    assert thin < thick
    assert abs(thin - PRIORS[4]) < abs(thick - PRIORS[4]), "thin evidence should sit nearer the prior"


def test_no_evidence_returns_the_prior():
    p = sft.calibrated_p(row(gate=4, n_precedents=0, n_precedents_cleared=0,
                             cluster_base_rate=None), PRIORS)
    assert p == pytest.approx(PRIORS[4], abs=0.01)


def test_probability_stays_inside_the_open_interval():
    for n, k in ((0, 0), (500, 500), (500, 0), (1, 0)):
        p = sft.calibrated_p(row(n_precedents=n, n_precedents_cleared=k, cluster_base_rate=None), PRIORS)
        assert 0.0 < p < 1.0


def test_difficult_features_lower_the_estimate():
    plain = sft.calibrated_p(row(gate=4), PRIORS)
    membrane = sft.calibrated_p(row(gate=4, tm_helices=7), PRIORS)
    assert membrane < plain


# --------------------------------------------------------------------------- formatting

def test_junk_organism_names_are_not_printed_as_species():
    """The archive writes 'Other' and 'unknown' in the organism field."""
    for junk in ("Other", "unknown", "", "N/A", "synthetic construct"):
        assert sft.clean_organism(junk) == ""
    assert sft.clean_organism("Escherichia coli") == "Escherichia coli"


def test_censored_precedents_are_labelled_as_such():
    rows = [{"target_id": "A-1", "centre": "X", "organism": "E. coli", "max_stage": 2, "censored": True}]
    assert "censored at centre closure" in sft.fmt_precedents(rows)


def test_feature_block_never_contains_the_sequence():
    r = row()
    r["sequence"] = "MKTAYIAKQRQISFVKSHFSRQ"
    block = sft.fmt_features(r)
    assert "MKTAYIAKQR" not in block


def test_gate_prompt_names_the_transition_not_just_the_stage():
    rec = sft.gate_judgement(row(gate=1), random.Random(0), PRIORS)
    assert "cloned -> expressed" in rec["messages"][1]["content"]


def test_gate_judgement_has_no_self_contradicting_verdict():
    rec = sft.gate_judgement(row(gate=0, label="failed", n_precedents=20,
                                 n_precedents_cleared=18), random.Random(0), PRIORS)
    body = rec["messages"][2]["content"]
    assert "Expected outcome" not in body
    p = float(re.search(r": ([0-9.]+)", body).group(1))
    assert p > 0.5, "the evidence says likely; the completion must not also say it stalls"


# --------------------------------------------------------------------------- the corpus

@needs_corpus
def test_every_record_is_valid_chat_json():
    with TRAIN.open() as fh:
        for i, line in enumerate(fh):
            if i > 3000:
                break
            rec = json.loads(line)
            assert set(rec) == {"messages"}
            roles = [m["role"] for m in rec["messages"]]
            assert roles == ["system", "user", "assistant"]
            assert all(m["content"].strip() for m in rec["messages"])


@needs_corpus
def test_no_completion_names_an_identifier_absent_from_its_own_prompt():
    """A hallucinated target ID is an automatic eval failure, so the corpus must never
    teach the habit."""
    pat = re.compile(r"\b[A-Z][A-Za-z0-9]{1,12}-[A-Za-z0-9_.]{2,15}\b")
    with TRAIN.open() as fh:
        for i, line in enumerate(fh):
            if i > 5000:
                break
            rec = json.loads(line)
            prompt = rec["messages"][1]["content"]
            for ident in pat.findall(rec["messages"][2]["content"]):
                assert ident in prompt, f"line {i}: {ident} invented"


@needs_corpus
def test_records_are_sorted_by_length_for_mlx_padding():
    lens = [len(json.loads(l)["messages"][1]["content"]) +
            len(json.loads(l)["messages"][2]["content"]) for l in TRAIN.open()]
    # sorted by an approximate token count, so allow the odd inversion within a bucket
    inversions = sum(1 for a, b in zip(lens, lens[1:]) if a > b + 40)
    assert inversions == 0, f"{inversions} records badly out of order"


@needs_corpus
def test_task_mix_matches_the_specification():
    rows = [json.loads(l) for l in TASKS.open() if l.strip()]
    n = len(rows)
    want = {"gate_judgement": 0.45, "pipeline_forecast": 0.20, "orthologue_ranking": 0.15,
            "construct_recommend": 0.12, "failure_attribution": 0.08}
    got = {}
    for r in rows:
        got[r["task"]] = got.get(r["task"], 0) + 1
    for k, w in want.items():
        assert abs(got.get(k, 0) / n - w) < 0.01, (k, got.get(k, 0) / n)


@needs_corpus
def test_no_censored_row_reached_the_corpus():
    rows = [json.loads(l) for l in TASKS.open() if l.strip()]
    assert not [r for r in rows if r.get("label") == "censored"]


@needs_corpus
def test_corpus_probabilities_are_calibrated():
    """The central claim: a model reproducing these targets is calibrated by construction."""
    rows = [json.loads(l) for l in TASKS.open() if l.strip()]
    gj = [r for r in rows if r["task"] == "gate_judgement" and r.get("p") is not None]
    assert len(gj) > 1000
    buckets: dict[float, list[bool]] = {}
    for r in gj:
        buckets.setdefault(round(r["p"], 1), []).append(r["label"] == "cleared")
    err = sum(abs(sum(v) / len(v) - k) * len(v) for k, v in buckets.items()) / len(gj)
    assert err < 0.05, f"expected calibration error of the training targets is {err:.3f}"
