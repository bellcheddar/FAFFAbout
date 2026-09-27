"""The 'what close relatives tried' panel: every statement must be true of the relatives shown,
and censored relatives must never count as evidence (rule 2)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import relatives as REL  # noqa: E402


def rel(tid, max_stage, censored=False, host="", tag="", trials=()):
    return {"target_id": tid, "max_stage": max_stage, "censored": censored, "identity": 0.5,
            "host": host, "tag": tag, "trials": list(trials)}


def trial(best, kind="", stop=""):
    return {"best": best, "construct_kind": kind, "stop_status": stop}


def test_clearance_counts_only_relatives_that_reached_the_gate():
    f = REL.facts([rel("A-1", 4), rel("A-2", 6), rel("A-3", 2)], gate=4)
    assert "1 of the 2 close relatives that reached purified got through to crystallised." in f


def test_censored_relatives_never_count_as_failures():
    f = REL.facts([rel("A-1", 4, censored=True), rel("A-2", 5)], gate=4)
    assert "The one close relative that reached purified got through to crystallised." in f
    assert any("closed at centre shutdown" in x and "left out of these counts" in x for x in f)


def test_single_and_zero_cases_read_as_english():
    assert "The one close relative that reached cloned did not get through to expressed." in \
        REL.facts([rel("A-1", 1)], gate=1)
    assert "None of the 2 close relatives that reached cloned got through to expressed." in \
        REL.facts([rel("A-1", 1), rel("A-2", 1)], gate=1)


def test_administrative_stop_reasons_are_not_reported_as_the_cause():
    trials = [trial(4, stop="other"), trial(4, stop="duplicate target found"),
              trial(4, stop="crystallization failed")]
    f = REL.facts([rel("A-1", 4, trials=trials)], gate=4)
    assert any('"crystallization failed" (1 trial)' in x for x in f)
    assert not any('"other"' in x for x in f)


def test_construct_comparison_names_the_one_that_got_further():
    trials = [trial(1, "full length ORF"), trial(4, "truncated ORF")]
    f = REL.facts([rel("A-1", 4, trials=trials)], gate=4)
    assert any("truncated or domain constructs got further (purified) than full-length ones (cloned)" in x for x in f)


def test_no_relatives_means_no_facts():
    assert REL.facts([], gate=3) == []


def test_construct_text_reads_boundaries_from_the_notes():
    assert REL._construct("truncated ORF", "MATCH|Sequence start: 12 Sequence end: 240 PCR") == \
        "truncated ORF, residues 12-240"
    assert REL._construct("", None) == ""


def test_display_puts_relatives_that_met_the_same_wall_before_closer_ones_that_did_not():
    """The DnaK case: six 80-100% relatives never cloned, a 78% one reached crystallised."""
    close = [dict(rel(f"C-{i}", 0), identity=0.9) for i in range(6)]
    useful = dict(rel("U-1", 5), identity=0.78)
    order = sorted(close + [useful], key=lambda r: REL.informative_first(r, 4))
    assert order[0]["target_id"] == "U-1"
