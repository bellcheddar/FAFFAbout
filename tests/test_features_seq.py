"""Sequence-feature tests, including cross-checks against Biopython.

A silent sign error in a hydropathy scale produces plausible output indefinitely, so the
composition features are checked against an independent implementation rather than only
against themselves.

Run:  .venv/bin/python -m pytest -q tests/test_features_seq.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import features_seq as fs  # noqa: E402

# Real sequences: T4 lysozyme (soluble, ordered) and a synthetic polytopic membrane protein.
LYSOZYME = ("MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILR"
            "NAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL")
POLY_A = "A" * 60
POLY_K = "K" * 40
POLY_E = "E" * 40
DISORDERED = "KRKRKRSPSPSPEEDDKRKRSPSPSPEEDDKRKRSPSPSPEEDDKRKRSPSPSPEEDD"


def counts_of(seq):
    return fs.counts_matrix([seq])


# --------------------------------------------------------------------------- composition

def test_gravy_matches_biopython():
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    for seq in (LYSOZYME, POLY_A, DISORDERED):
        mine = float(fs.gravy(counts_of(seq))[0])
        theirs = ProteinAnalysis(seq).gravy()
        assert mine == pytest.approx(theirs, abs=1e-6), seq[:20]


def test_aromatic_fraction_matches_biopython():
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    mine = float(fs.aromatic_fraction(counts_of(LYSOZYME))[0])
    theirs = ProteinAnalysis(LYSOZYME).aromaticity()
    assert mine == pytest.approx(theirs, abs=1e-6)


def test_isoelectric_point_is_close_to_biopython():
    """Different pKa sets shift pI by a few tenths, so this checks agreement not identity."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    mine = float(fs.isoelectric_point(counts_of(LYSOZYME))[0])
    theirs = ProteinAnalysis(LYSOZYME).isoelectric_point()
    assert mine == pytest.approx(theirs, abs=0.6), (mine, theirs)
    assert 8.0 < mine < 11.0, "T4 lysozyme is a basic protein"


def test_polybasic_and_polyacidic_pi_are_at_the_extremes():
    assert float(fs.isoelectric_point(counts_of(POLY_K))[0]) > 10.0
    assert float(fs.isoelectric_point(counts_of(POLY_E))[0]) < 4.5


def test_net_charge_signs_are_the_right_way_round():
    assert float(fs.net_charge(counts_of(POLY_K))[0]) > 30
    assert float(fs.net_charge(counts_of(POLY_E))[0]) < -30


def test_charge_is_zero_at_the_isoelectric_point():
    for seq in (LYSOZYME, DISORDERED, POLY_K):
        c = counts_of(seq)
        pi = fs.isoelectric_point(c)
        assert float(fs.net_charge_at(c, pi)[0]) == pytest.approx(0.0, abs=1e-3), seq[:15]


def test_residue_fraction_counts_correctly():
    assert float(fs.residue_fraction(counts_of("ACACAC"), "C")[0]) == pytest.approx(0.5)
    assert float(fs.residue_fraction(counts_of(POLY_A), "C")[0]) == 0.0


def test_non_standard_residues_are_ignored_not_counted():
    """Selenomethionine and X appear in the archive; they must not shift a fraction."""
    assert fs.counts_matrix(["ACDXU"])[0].sum() == 3
    assert float(fs.residue_fraction(fs.counts_matrix(["ACDXU"]), "A")[0]) == pytest.approx(1 / 3)


def test_empty_sequence_gives_nan_not_a_crash():
    c = fs.counts_matrix([""])
    assert np.isnan(fs.gravy(c)[0])
    assert np.isnan(fs.isoelectric_point(c)[0])


# --------------------------------------------------------------------------- topology

def test_soluble_protein_has_no_predicted_tm_helix():
    assert fs.tm_helices(LYSOZYME) == 0


def test_single_membrane_span_is_found():
    seq = "MKKTAIAIAVALAGFATVAQA" + "LLIVGGVVLLLGVLAFLILGWVLAGL" + "DKQRTEEHNGKEEDR"
    assert fs.tm_helices(seq) >= 1


def test_multiple_spans_are_counted_separately():
    span = "LLIVGGVVLLLGVLAFLILGWVLAGLIL"
    linker = "DKQRTEEHNGKEEDRKKRDDEEKKRDDEE"
    assert fs.tm_helices(span + linker + span + linker + span) >= 3


def test_signal_peptide_detected_only_at_the_n_terminus():
    signal = "MKKTAIAIAVALAGFATVAQA"
    tail = "DKQRTEEHNGKEEDRKKRDDEE" * 3
    assert fs.has_signal_peptide(signal + tail) is True
    # the same hydrophobic stretch far from the N-terminus is not a signal peptide
    assert fs.has_signal_peptide(tail + signal) is False


# --------------------------------------------------------------------------- disorder

def test_foldindex_calls_a_charged_repeat_disordered():
    overall, _, _, _ = fs.disorder_fractions(DISORDERED)
    assert overall > 0.5


def test_foldindex_calls_a_folded_protein_ordered():
    overall, _, _, _ = fs.disorder_fractions(LYSOZYME)
    assert overall < 0.2


def test_terminal_disorder_is_reported_where_it_is():
    seq = "KRKRSPSPEEDDKRKRSPSPEEDD" * 3 + LYSOZYME
    _, nterm, cterm, _ = fs.disorder_fractions(seq)
    assert nterm > cterm, "the disordered stretch is at the N-terminus"


def test_disorder_fractions_are_all_fractions():
    for seq in (LYSOZYME, DISORDERED, POLY_A):
        for v in fs.disorder_fractions(seq):
            assert 0.0 <= v <= 1.0 or np.isnan(v)


def test_short_sequence_does_not_crash_the_profile():
    assert fs.foldindex_profile("MKT").size >= 0
    assert all(np.isnan(v) or 0 <= v <= 1 for v in fs.disorder_fractions("MKT"))


# --------------------------------------------------------------------------- complexity

def test_homopolymer_is_entirely_low_complexity():
    assert fs.low_complexity_fraction(POLY_A) == pytest.approx(1.0)


def test_diverse_sequence_is_not_low_complexity():
    assert fs.low_complexity_fraction(LYSOZYME) < 0.1


def test_sequence_shorter_than_the_window_is_zero_not_an_error():
    assert fs.low_complexity_fraction("MKTAYIAK") == 0.0


# --------------------------------------------------------------------------- vectorisation

def test_batch_matches_one_at_a_time():
    seqs = [LYSOZYME, POLY_A, DISORDERED, POLY_K]
    batch = fs.counts_matrix(seqs)
    for i, s in enumerate(seqs):
        assert np.array_equal(batch[i], fs.counts_matrix([s])[0])
    assert np.allclose(fs.gravy(batch), [float(fs.gravy(counts_of(s))[0]) for s in seqs],
                       equal_nan=True)
    assert np.allclose(fs.isoelectric_point(batch),
                       [float(fs.isoelectric_point(counts_of(s))[0]) for s in seqs], equal_nan=True)
