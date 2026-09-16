"""A sequence with no scoreable residues must be refused, not forecast from.

The bug this pins down, found on 2026-09-16 while chasing an unrelated evaluation result:

    resolve.from_fasta(">t\\n" + "X"*30)   ->  accepted
    sequence_features("X"*30)              ->  EIGHT NaN features

`AA_OK` admits X, B, Z, U, O and *, none of which carries a value on any composition
scale, so a sequence made only of those divides by zero in pI, GRAVY, net charge and all
five composition fractions. NaN does not raise when formatted, it RENDERS, so the user got
a 200 with "pI: nan" in the prompt while the booster received 11 of its 37 features as
missing and answered from a learned default split: a confident forecast built on no
sequence information at all.

The boundary is exact. ONE standard residue among thirty-nine X's makes every scale
finite, so this is a single empty-denominator case and not a general problem with masked
sequences. The tests below pin both sides of that boundary, because a guard that rejected
masked sequences generally would be a regression, not a fix.

Run:  .venv/bin/python -m pytest -q tests/test_nan_boundary.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import llm  # noqa: E402
import predict  # noqa: E402
import resolve as R  # noqa: E402

FEATS = getattr(predict, "sequence_features", None) or getattr(predict, "seq_features")


# --------------------------------------------------------------- the boundary itself

def test_no_standard_residues_is_refused():
    with pytest.raises(R.ResolveError):
        R.from_fasta(">t\n" + "X" * 30)


def test_one_standard_residue_is_enough_to_be_scoreable():
    """The fix must not reject masked sequences generally: 39 X's and one M is finite."""
    r = R.from_fasta(">t\n" + "M" + "X" * 39)
    f = FEATS(r.sequence)
    bad = [k for k, v in f.items() if isinstance(v, float) and not math.isfinite(v)]
    assert bad == [], f"expected all-finite features, got NaN/inf in {bad}"


@pytest.mark.parametrize("seq", ["X" * 30, "B" * 30, "Z" * 30, "U" * 30, "XBZUO" * 6])
def test_every_placeholder_only_sequence_is_refused(seq):
    with pytest.raises(R.ResolveError):
        R.from_fasta(">t\n" + seq)


def test_n_standard_counts_only_the_twenty():
    assert R.n_standard("XXXX") == 0
    assert R.n_standard("MGKII") == 5
    assert R.n_standard("MXGXK") == 3


# --------------------------------------------------------------- all three entry points

def test_assert_scoreable_is_reachable_from_every_route():
    """from_uniprot and from_pdb build Resolved directly and bypass from_fasta. A PDB
    chain of UNK residues maps to poly-X, which is a routine low-resolution model rather
    than a pathological paste, so guarding from_fasta alone would have missed it."""
    src = (ROOT / "app" / "resolve.py").read_text()
    body = src.split("def from_fasta", 1)[1]
    for route in ("from_fasta", "from_uniprot", "from_pdb"):
        seg = src.split(f"def {route}", 1)[1].split("\ndef ", 1)[0]
        assert "assert_scoreable" in seg, f"{route} does not guard against unscoreable input"
    assert body  # the split above must actually have found from_fasta


def test_assert_scoreable_passes_a_real_sequence():
    R.assert_scoreable("MGKIIGIDLGTTNSCVAIMDGTTNSCVAIM")


# --------------------------------------------------------------- rendering, not raising

def test_finite_rejects_nan_none_and_bool():
    assert llm._finite(0.07) is True
    assert llm._finite(0) is True
    assert llm._finite(float("nan")) is False
    assert llm._finite(float("inf")) is False
    assert llm._finite(None) is False
    # bool is an int subclass, and True would otherwise format as a percentage
    assert llm._finite(True) is False


def nan_payload() -> dict:
    nan = float("nan")
    return {
        "target": {"accession": "P0", "organism": "Escherichia coli", "length": 30,
                   "source": "fasta", "name": "t", "chain": "", "taxon_id": "",
                   "sequence": "X" * 30, "warnings": []},
        "features": {"length": 30, "pI": nan, "gravy": nan, "net_charge_ph7": nan,
                     "cys_count": 0, "met_count": 0, "tm_helices": 0, "signal_peptide": False,
                     "disorder_frac": nan, "disorder_nterm": nan, "disorder_cterm": nan,
                     "low_complexity_frac": nan, "superkingdom": "Bacteria", "genus": "E"},
        "ladder": ["selected", "cloned", "expressed", "soluble", "purified",
                   "crystallised", "diffracting", "structure", "deposited"],
        "conditional": [], "survival": [],
        "bottleneck": {"gate": 0, "name": "selected", "next": "cloned", "action": "x",
                       "drop": 0.1, "conditional": 0.9, "interval": [0.1, 0.2]},
        "evidence": {"n_precedents": 5, "n_close": 1, "n_censored": 0,
                     "closest_identity": 0.5, "strength": "WEAK", "per_gate": {}},
        "precedents": [], "counterfactuals": [], "caveats": [],
        "model": {"probabilities_from": "x", "narrative_from": None, "narrative": None},
    }


def test_no_nan_reaches_the_prompt_by_any_route():
    """Defence in depth behind the resolve guard.

    The first pass at this grepped for ":.0%" and so guarded the disorder lines while
    leaving pI, GRAVY and net charge on a bare {f['pI']}, which formats NaN just as
    silently. A live check found three lines still reading "nan". Fixing only the sites a
    search pattern happens to match is how a partial fix passes for a complete one.
    """
    prompt = llm.build_prompt(nan_payload())
    offenders = [ln.strip() for ln in prompt.splitlines() if "nan" in ln.lower()]
    assert offenders == [], f"NaN rendered into the prompt: {offenders}"


def test_the_prompt_is_still_useful_without_those_features():
    """Omitting must not empty the block: what IS known still has to reach the model."""
    prompt = llm.build_prompt(nan_payload())
    assert "Target features:" in prompt
    assert "length: 30 residues" in prompt
    assert "Archive evidence:" in prompt
