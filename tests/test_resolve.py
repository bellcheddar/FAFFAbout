"""Input-resolution tests.

A silently mis-resolved identifier produces a confident forecast for the wrong protein, so
sniffing and cleaning are tested on the shapes people actually paste.

The two network tests are marked and skipped by default: run them with `-m network`.

Run:  .venv/bin/python -m pytest -q tests/test_resolve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import resolve as R  # noqa: E402

LYSOZYME = ("MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILR"
            "NAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL")


# --------------------------------------------------------------------------- sniffing

@pytest.mark.parametrize("text,kind", [
    (">sp|P0A6Y8|DNAK_ECOLI\nMGKIIGIDLGTTNS", "fasta"),
    (LYSOZYME, "fasta"),
    ("  " + LYSOZYME + "\n", "fasta"),
    ("P0A6Y8", "uniprot"),
    ("p0a6y8", "uniprot"),
    ("P0A6Y8-2", "uniprot"),
    ("A0A023GPI8", "uniprot"),
    ("4XB7", "pdb"),
    ("4xb7_A", "pdb"),
    ("4XB7.A", "pdb"),
    ("4XB7:B", "pdb"),
])
def test_sniff(text, kind):
    assert R.sniff(text) == kind


@pytest.mark.parametrize("bad", ["", "   ", "hello there", "12345", "???"])
def test_sniff_rejects_what_it_cannot_identify(bad):
    with pytest.raises(R.ResolveError):
        R.sniff(bad)


def test_a_pdb_id_is_not_mistaken_for_a_sequence():
    """4XB7 is four characters of which three are valid residues."""
    assert R.sniff("4XB7") == "pdb"


def test_a_short_peptide_is_rejected_rather_than_guessed():
    with pytest.raises(R.ResolveError):
        R.resolve("MKTAYIAK")


# --------------------------------------------------------------------------- cleaning

def test_numbered_and_wrapped_sequence_is_cleaned():
    pasted = "1 MNIFEMLRID EGLRLKIYKD\n21 TEGYYTIGIG HLLTKSPSLN"
    seq, warnings = R.clean_sequence(pasted)
    assert seq == "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLN"
    assert not warnings


def test_alignment_gaps_are_stripped():
    seq, _ = R.clean_sequence("MKT--AYIA..KQR")
    assert seq == "MKTAYIAKQR"


def test_unrecognised_characters_are_dropped_and_reported():
    seq, warnings = R.clean_sequence("MKT@AYI#AKQR")
    assert seq == "MKTAYIAKQR"
    assert warnings and "unrecognised" in warnings[0]


def test_selenocysteine_is_kept_but_flagged():
    seq, warnings = R.clean_sequence("MKTUAYIAKQR")
    assert "U" in seq
    assert any("selenocysteine" in w for w in warnings)


def test_fasta_header_is_captured_not_treated_as_sequence():
    r = R.from_fasta(">sp|P0A6Y8|DNAK_ECOLI Chaperone protein DnaK\n" + LYSOZYME)
    assert r.sequence == LYSOZYME
    assert "DnaK" in r.header
    assert r.source == "fasta"


def test_lowercase_sequence_is_uppercased():
    r = R.from_fasta(LYSOZYME.lower())
    assert r.sequence == LYSOZYME


# --------------------------------------------------------------------------- network

@pytest.mark.network
def test_uniprot_resolves_dnak():
    r = R.from_uniprot("P0A6Y8")
    assert r.length == 638
    assert r.sequence.startswith("MGKIIGIDLGTTNS")
    assert "DnaK" in r.name
    assert r.taxon_id == "83333"


@pytest.mark.network
def test_pdb_resolves_a_chain_and_warns_about_the_construct():
    r = R.from_pdb("4XB7_A")
    assert r.length > 100
    assert r.source == "pdb" and r.accession == "4XB7"
    assert any("construct that crystallised" in w for w in r.warnings)


@pytest.mark.network
def test_an_unknown_accession_is_an_error_not_an_empty_sequence():
    with pytest.raises(R.ResolveError):
        R.from_uniprot("P00000")
