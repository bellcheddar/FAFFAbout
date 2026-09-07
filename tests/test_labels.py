"""Label-set tests: the four negative-mining rules, expressed as assertions.

Run:  .venv/bin/python -m pytest -q tests/test_labels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

labels = __import__("06_build_labels")

REAL = ROOT / "data" / "parquet" / "labels_l1" / "l1.parquet"
needs_archive = pytest.mark.skipif(not REAL.exists(), reason="needs the built archive")


def make_db(tmp_path, rows):
    """rows: (target_id, centre, max_stage, censored, seq_md5, cluster30, cluster70)"""
    con = duckdb.connect()
    con.execute("CREATE TABLE targets (target_id VARCHAR, seq_md5 VARCHAR, organism VARCHAR, seq_len INTEGER)")
    con.execute("CREATE TABLE censoring (target_id VARCHAR, centre VARCHAR, max_stage TINYINT, "
                "method VARCHAR, censored BOOLEAN, censored_reason VARCHAR)")
    con.execute("CREATE TABLE clusters_30 (seq_md5 VARCHAR, cluster_id INTEGER)")
    con.execute("CREATE TABLE clusters_70 (seq_md5 VARCHAR, cluster_id INTEGER)")
    for tid, centre, ms, cens, md5, c30, c70 in rows:
        con.execute("INSERT INTO targets VALUES (?, ?, 'E. coli', 300)", [tid, md5])
        con.execute("INSERT INTO censoring VALUES (?, ?, ?, 'xray', ?, ?)",
                    [tid, centre, ms, cens, "bulk_closure" if cens else ""])
        con.execute("INSERT INTO clusters_30 VALUES (?, ?)", [md5, c30])
        if c70 is not None:
            con.execute("INSERT INTO clusters_70 VALUES (?, ?)", [md5, c70])
    labels.PQ = tmp_path
    return con


def l1_rows(con):
    return con.execute("SELECT target_id, gate, label, in_loss FROM labels_l1 ORDER BY target_id, gate").fetchall()


# --------------------------------------------------------------------------- L1 shape

def test_gates_never_entered_are_not_emitted(tmp_path):
    """Not-attempted is not a negative: a target that died at `selected` says nothing
    about crystallisation, so no crystallisation row exists for it."""
    con = make_db(tmp_path, [("A", "X", 1, False, "m1", 1, 1)])
    labels.build_l1(con)
    assert [(g, lab) for _, g, lab, _ in l1_rows(con)] == [(0, "cleared"), (1, "failed")]


def test_deposited_target_clears_all_eight_gates_and_none_say_failed(tmp_path):
    """The bug this guards: applying the spec's rule literally at gate 8 labels every
    successful target `failed` at the final gate."""
    con = make_db(tmp_path, [("A", "X", 8, False, "m1", 1, 1)])
    labels.build_l1(con)
    rows = l1_rows(con)
    assert len(rows) == 8
    assert {lab for _, _, lab, _ in rows} == {"cleared"}
    assert max(g for _, g, _, _ in rows) == 7, "there is no gate beyond deposition"


def test_uncensored_target_fails_at_exactly_its_terminal_gate(tmp_path):
    con = make_db(tmp_path, [("A", "X", 4, False, "m1", 1, 1)])
    labels.build_l1(con)
    assert [(g, lab) for _, g, lab, _ in l1_rows(con)] == [
        (0, "cleared"), (1, "cleared"), (2, "cleared"), (3, "cleared"), (4, "failed")]


# --------------------------------------------------------------------------- censoring

def test_censored_target_keeps_its_cleared_rows_and_loses_only_the_terminal_one(tmp_path):
    """A censored target passed real gates. Only its final gate is unknown."""
    con = make_db(tmp_path, [("A", "X", 4, True, "m1", 1, 1)])
    labels.build_l1(con)
    rows = l1_rows(con)
    assert [(g, lab, in_loss) for _, g, lab, in_loss in rows] == [
        (0, "cleared", True), (1, "cleared", True), (2, "cleared", True),
        (3, "cleared", True), (4, "censored", False)]


def test_no_censored_row_ever_enters_the_loss(tmp_path):
    con = make_db(tmp_path, [(f"T{i}", "X", i % 8, i % 2 == 0, f"m{i}", 1, 1) for i in range(16)])
    labels.build_l1(con)
    assert con.execute("SELECT count(*) FROM labels_l1 WHERE label = 'censored' AND in_loss").fetchone()[0] == 0


def test_l2_excludes_censored_targets_entirely(tmp_path):
    con = make_db(tmp_path, [("KEEP", "X", 8, False, "m1", 1, 1),
                             ("DROP", "X", 3, True, "m2", 1, 1)])
    labels.build_l2(con)
    ids = [r[0] for r in con.execute("SELECT target_id FROM labels_l2").fetchall()]
    assert ids == ["KEEP"]


def test_l2_outcome_names_the_stalling_gate(tmp_path):
    con = make_db(tmp_path, [("A", "X", 8, False, "m1", 1, 1), ("B", "X", 3, False, "m2", 1, 1)])
    labels.build_l2(con)
    got = dict(con.execute("SELECT target_id, outcome FROM labels_l2").fetchall())
    assert got == {"A": "deposited", "B": "stalled_at_3"}


# --------------------------------------------------------------------------- L3 pairs

def test_l3_requires_a_gap_of_more_than_one_gate(tmp_path):
    """Adjacent fates are noise; the pair has to disagree by more than one rung."""
    con = make_db(tmp_path, [("A", "X", 5, False, "m1", 7, 70),
                             ("B", "X", 4, False, "m2", 7, 70),   # gap 1: excluded
                             ("C", "X", 2, False, "m3", 7, 70)])  # gap 3 vs A: included
    labels.build_l3(con, cap=20)
    pairs = con.execute("SELECT chosen_id, rejected_id, stage_gap FROM labels_l3 ORDER BY 1, 2").fetchall()
    assert ("A", "B", 1) not in pairs
    assert ("A", "C", 3) in pairs


def test_l3_marks_pairs_sharing_a_70pct_cluster_as_hard(tmp_path):
    con = make_db(tmp_path, [("A", "X", 6, False, "m1", 7, 70),
                             ("B", "X", 1, False, "m2", 7, 70),    # same 70% cluster: hard
                             ("C", "X", 1, False, "m3", 7, 99)])   # same 30% only: not hard
    labels.build_l3(con, cap=20)
    got = dict(((c, r), h) for c, r, h in
               con.execute("SELECT chosen_id, rejected_id, hard FROM labels_l3").fetchall())
    assert got[("A", "B")] is True
    assert got[("A", "C")] is False


def test_l3_never_pairs_a_censored_target(tmp_path):
    con = make_db(tmp_path, [("A", "X", 6, False, "m1", 7, 70),
                             ("B", "X", 1, True, "m2", 7, 70)])
    labels.build_l3(con, cap=20)
    assert con.execute("SELECT count(*) FROM labels_l3").fetchone()[0] == 0


def test_l3_cap_limits_pairs_per_cluster(tmp_path):
    rows = [("HI", "X", 7, False, "mh", 7, 70)]
    rows += [(f"LO{i}", "X", 1, False, f"m{i}", 7, 70) for i in range(30)]
    con = make_db(tmp_path, rows)
    labels.build_l3(con, cap=5)
    assert con.execute("SELECT count(*) FROM labels_l3").fetchone()[0] == 5


def test_l3_cap_keeps_the_hard_pairs_first(tmp_path):
    """The cap must not discard the highest-value stratum for an arbitrary slice."""
    rows = [("HI", "X", 7, False, "mh", 7, 70)]
    rows += [(f"EASY{i}", "X", 1, False, f"e{i}", 7, 900 + i) for i in range(10)]
    rows += [(f"HARD{i}", "X", 1, False, f"h{i}", 7, 70) for i in range(3)]
    con = make_db(tmp_path, rows)
    labels.build_l3(con, cap=3)
    kept = [r[0] for r in con.execute("SELECT rejected_id FROM labels_l3").fetchall()]
    assert sorted(kept) == ["HARD0", "HARD1", "HARD2"]


# --------------------------------------------------------------------------- the real archive

@needs_archive
def test_real_l1_has_no_gate_8_rows():
    con = duckdb.connect(str(ROOT / "data" / "faffabout.duckdb"), read_only=True)
    assert con.execute("SELECT count(*) FROM labels_l1 WHERE gate > 7").fetchone()[0] == 0


@needs_archive
def test_real_l1_never_labels_a_deposited_target_failed():
    con = duckdb.connect(str(ROOT / "data" / "faffabout.duckdb"), read_only=True)
    assert con.execute(
        "SELECT count(*) FROM labels_l1 WHERE max_stage >= 8 AND label <> 'cleared'").fetchone()[0] == 0


@needs_archive
def test_real_censored_rows_are_exactly_one_per_censored_target():
    con = duckdb.connect(str(ROOT / "data" / "faffabout.duckdb"), read_only=True)
    held = con.execute("SELECT count(*) FROM labels_l1 WHERE NOT in_loss").fetchone()[0]
    cens = con.execute("SELECT count(*) FROM censoring WHERE censored").fetchone()[0]
    assert held == cens, "each censored target should lose exactly its terminal gate row"
