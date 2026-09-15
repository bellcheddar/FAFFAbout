"""Tests for the narrative path, which has never run against a real model.

`app/llm.py` is the only place a language model touches the product, and `sanitise` is the
single thing standing between a hallucinated precedent identifier and a reader. No adapter
exists yet, so none of this code has ever executed: that is exactly when to test it, not
after it has been trusted in production for a week.

Nothing here needs a model. `narrate` is the only function that makes a request, and it is
not exercised.

Run:  .venv/bin/python -m pytest -q tests/test_llm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import llm  # noqa: E402


def payload(**over) -> dict:
    d = {
        "target": {"accession": "P0A6Y8", "organism": "Escherichia coli", "length": 638,
                   "source": "uniprot", "name": "Chaperone protein DnaK", "chain": "",
                   "taxon_id": "83333", "sequence": "MGKIIGIDLGTTNSCVAIMDG", "warnings": []},
        "features": {"length": 638, "pI": 4.58, "gravy": -0.409, "net_charge_ph7": -24.1,
                     "cys_count": 1, "met_count": 12, "tm_helices": 0, "signal_peptide": False,
                     "disorder_frac": 0.07, "disorder_nterm": 0.07, "disorder_cterm": 0.02,
                     "low_complexity_frac": 0.03, "superkingdom": "Bacteria", "genus": "Escherichia"},
        "ladder": ["selected", "cloned", "expressed", "soluble", "purified",
                   "crystallised", "diffracting", "structure", "deposited"],
        "conditional": [0.99, 0.99, 0.99, 0.99, 0.18, 0.95, 0.56, 0.97],
        "survival": [1.0, 0.99, 0.98, 0.97, 0.96, 0.18, 0.17, 0.09, 0.09],
        "bottleneck": {"gate": 4, "name": "purified", "next": "crystallised",
                       "action": "obtaining crystals", "drop": 0.78, "conditional": 0.18,
                       "interval": [0.112, 0.28]},
        "evidence": {"n_precedents": 297, "n_close": 14, "n_censored": 174,
                     "closest_identity": 1.0, "strength": "STRONG", "per_gate": {}},
        "precedents": [
            {"target_id": "MCSG-APC106035", "centre": "MCSG", "organism": "Escherichia coli CFT073",
             "identity": 0.998, "max_stage": 0, "censored": False, "note": ""},
            {"target_id": "CESG-GO.97406", "centre": "CESG", "organism": "Escherichia coli",
             "identity": 0.991, "max_stage": 0, "censored": True, "note": "censored"},
        ],
        "counterfactuals": [], "caveats": [],
        "model": {"probabilities_from": "lightgbm:declared_", "narrative_from": None, "narrative": None},
    }
    d.update(over)
    return d


# --------------------------------------------------------------------------- the prompt

def test_prompt_never_contains_the_raw_sequence():
    """Spec 5.4: the residues never go in. Llama's tokeniser shreds them and they teach
    the model nothing, which is the whole reason the feature block exists."""
    p = llm.build_prompt(payload())
    assert "MGKIIGIDLGTTNSCVAIMDG" not in p


def test_prompt_carries_the_precedent_identifiers():
    """The model must cite what it was given, so it has to be given it."""
    p = llm.build_prompt(payload())
    assert "MCSG-APC106035" in p
    assert "CESG-GO.97406" in p


def test_prompt_flags_censored_precedents():
    p = llm.build_prompt(payload())
    assert "censored at centre closure" in p


def test_prompt_states_the_computed_numbers_as_given():
    p = llm.build_prompt(payload())
    assert "purified -> crystallised" in p
    assert "are not yours to change" in p


def test_prompt_survives_a_target_with_no_precedents():
    p = llm.build_prompt(payload(precedents=[]))
    assert "Precedents:" not in p
    assert "purified" in p


# --------------------------------------------------------------------------- the guard

def test_an_invented_target_identifier_is_removed():
    """The automatic-fail condition in eval_generative.py. The serving path will not pass
    one to a reader even if the model produces it."""
    d = payload()
    prompt = llm.build_prompt(d)
    text = "The closest precedent is JCSG-999999, which crystallised."
    clean, removed = llm.sanitise(text, prompt, d)
    assert "JCSG-999999" not in clean
    assert removed == ["JCSG-999999"]
    assert "identifier removed" in clean


def test_a_real_identifier_from_the_prompt_is_kept():
    d = payload()
    prompt = llm.build_prompt(d)
    text = "MCSG-APC106035 is the nearest relative at 100% identity."
    clean, removed = llm.sanitise(text, prompt, d)
    assert "MCSG-APC106035" in clean
    assert removed == []


def test_an_invented_pdb_code_is_removed():
    d = payload()
    prompt = llm.build_prompt(d)
    clean, removed = llm.sanitise("See 4XB7 for the fold.", prompt, d)
    assert "4XB7" not in clean
    assert "4XB7" in removed


def test_ordinary_prose_is_left_alone():
    d = payload()
    prompt = llm.build_prompt(d)
    text = ("Crystallisation is the wall here: the conditional is 0.18 and the interval is "
            "wide because 174 of the retrieved records are censored.")
    clean, removed = llm.sanitise(text, prompt, d)
    assert clean == text
    assert removed == []


def test_a_removal_is_reported_as_a_caveat_not_hidden():
    d = payload()
    prompt = llm.build_prompt(d)
    text = "Compare with NESG-FAKE123."
    clean, removed = llm.sanitise(text, prompt, d)
    assert removed
    # narrate() appends the caveat; check the contract sanitise offers it
    assert "[identifier removed" in clean


@pytest.mark.parametrize("year", ["2017", "2000", "1999"])
def test_a_bare_year_is_not_mistaken_for_a_pdb_code(year):
    d = payload()
    prompt = llm.build_prompt(d)
    clean, _ = llm.sanitise(f"The archive froze in {year}.", prompt, d)
    assert year in clean


# --------------------------------------------------------------------------- degradation

def test_no_server_means_no_narrative_rather_than_an_error(monkeypatch):
    """A forecast is complete without prose. It must never fail because of the model."""
    monkeypatch.setattr(llm, "_MODELS_CACHE", [])
    monkeypatch.setattr(llm, "models", lambda: [])
    assert llm.available() is False
    assert llm.narrate(payload()) is None


def test_resolved_model_prefers_what_the_server_reports(monkeypatch):
    """The model field must match the server's resolved path; a mismatch 404s as an opaque
    hub-lookup error rather than a clear one."""
    monkeypatch.setattr(llm, "MODEL_NAME", "")
    monkeypatch.setattr(llm, "models", lambda: ["/Users/x/adapters/faffabout-round01"])
    assert llm.resolved_model() == "/Users/x/adapters/faffabout-round01"
