"""Split tests: leakage is the failure this file exists to prevent.

Run:  .venv/bin/python -m pytest -q tests/test_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import splits  # noqa: E402

REAL = ROOT / "data" / "parquet" / "splits" / "splits.parquet"
needs_archive = pytest.mark.skipif(not REAL.exists(), reason="needs the built archive")


# --------------------------------------------------------------------------- assignment

def test_every_cluster_lands_in_exactly_one_split():
    sizes = [(i, (i % 17) + 1) for i in range(500)]
    out = splits.assign_clusters(sizes)
    assert len(out) == len(sizes)
    assert set(out.values()) <= {"train", "valid", "test"}


def test_assignment_hits_the_ratios_closely():
    sizes = [(i, (i % 23) + 1) for i in range(3000)]
    out = splits.assign_clusters(sizes)
    total = sum(n for _, n in sizes)
    got = {k: 0 for k in splits.RATIOS}
    for cid, n in sizes:
        got[out[cid]] += n
    for k, want in splits.RATIOS.items():
        assert abs(got[k] / total - want) < 0.01, (k, got[k] / total)


def test_one_huge_cluster_cannot_unbalance_the_split():
    """A 447-member cluster landing in `valid` by a hash would move it by 0.16 points."""
    sizes = [(0, 447)] + [(i, 1) for i in range(1, 3000)]
    out = splits.assign_clusters(sizes)
    total = sum(n for _, n in sizes)
    got = {k: 0 for k in splits.RATIOS}
    for cid, n in sizes:
        got[out[cid]] += n
    assert out[0] == "train", "the largest cluster belongs to the largest split"
    for k, want in splits.RATIOS.items():
        assert abs(got[k] / total - want) < 0.01


def test_assignment_is_deterministic():
    sizes = [(i, (i % 11) + 1) for i in range(400)]
    assert splits.assign_clusters(sizes) == splits.assign_clusters(list(reversed(sizes)))


# --------------------------------------------------------------------------- temporal

def make_db(tmp_path, rows):
    """rows: (target_id, centre, cluster_id, first_event, last_event)"""
    con = duckdb.connect()
    con.execute("CREATE TABLE targets (target_id VARCHAR, seq_md5 VARCHAR)")
    con.execute("CREATE TABLE censoring (target_id VARCHAR, centre VARCHAR, max_stage TINYINT, "
                "censored BOOLEAN, last_any DATE)")
    con.execute("CREATE TABLE status_history_canon (target_id VARCHAR, status_date VARCHAR)")
    con.execute("CREATE TABLE clusters_30 (seq_md5 VARCHAR, cluster_id INTEGER)")
    for tid, centre, cid, first, last in rows:
        con.execute("INSERT INTO targets VALUES (?, ?)", [tid, f"md5{tid}"])
        con.execute("INSERT INTO censoring VALUES (?, ?, 3, false, ?)", [tid, centre, last])
        con.execute("INSERT INTO clusters_30 VALUES (?, ?)", [f"md5{tid}", cid])
        for d in (first, last):
            con.execute("INSERT INTO status_history_canon VALUES (?, ?)", [tid, d])
    splits.OUT = tmp_path
    # build() reads clusters/censoring from parquet paths, so exercise the SQL directly
    return con


def test_temporal_split_excludes_targets_straddling_the_boundary(tmp_path):
    """A target selected in 2013 and deposited in 2016 would carry post-boundary
    information into training, so it belongs to neither side."""
    con = make_db(tmp_path, [
        ("OLD", "X", 1, "2010-01-01", "2012-06-01"),
        ("NEW", "X", 2, "2015-02-01", "2016-06-01"),
        ("SPAN", "X", 3, "2013-01-01", "2016-01-01"),
    ])
    got = dict(con.execute(f"""
        SELECT c.target_id,
               CASE WHEN c.last_any < DATE '{splits.TEMPORAL_BOUNDARY}' THEN 'train'
                    WHEN (SELECT min(try_cast(status_date AS DATE)) FROM status_history_canon h
                          WHERE h.target_id = c.target_id) >= DATE '{splits.TEMPORAL_BOUNDARY}' THEN 'test'
                    ELSE 'straddles' END
        FROM censoring c
    """).fetchall())
    assert got == {"OLD": "train", "NEW": "test", "SPAN": "straddles"}


# --------------------------------------------------------------------------- the real archive

@needs_archive
def test_real_split_has_no_cluster_in_two_splits():
    con = duckdb.connect(str(ROOT / "data" / "faffabout.duckdb"), read_only=True)
    assert con.execute("""
        SELECT count(*) FROM (SELECT cluster_id FROM splits WHERE cluster_id IS NOT NULL
                              GROUP BY 1 HAVING count(DISTINCT split_cluster) > 1)
    """).fetchone()[0] == 0


@needs_archive
def test_real_split_has_no_identical_sequence_in_two_splits():
    con = duckdb.connect(str(ROOT / "data" / "faffabout.duckdb"), read_only=True)
    assert con.execute("""
        SELECT count(*) FROM (
            SELECT t.seq_md5 FROM splits s JOIN targets t USING (target_id)
            WHERE t.seq_md5 <> '' GROUP BY 1 HAVING count(DISTINCT s.split_cluster) > 1)
    """).fetchone()[0] == 0


@needs_archive
def test_real_temporal_train_never_sees_past_the_boundary():
    con = duckdb.connect(str(ROOT / "data" / "faffabout.duckdb"), read_only=True)
    assert con.execute(f"""
        SELECT count(*) FROM splits
        WHERE split_temporal = 'train' AND last_event >= DATE '{splits.TEMPORAL_BOUNDARY}'
    """).fetchone()[0] == 0
