"""Censoring tests: the rule the whole project rests on, exercised on synthetic centres.

Run:  .venv/bin/python -m pytest -q tests/test_censoring.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import censoring  # noqa: E402

FREEZE = "2017-06-30"


def make_db(tmp_path, rows):
    """rows: (target_id, centre, max_stage, [(status_canon, stage_ord, date), ...])"""
    con = duckdb.connect()
    con.execute("CREATE TABLE targets (target_id VARCHAR, centre VARCHAR)")
    con.execute("CREATE TABLE target_stage (target_id VARCHAR, centre VARCHAR, max_stage TINYINT, method VARCHAR)")
    con.execute("CREATE TABLE status_history_canon (target_id VARCHAR, centre VARCHAR, "
                "status_canon VARCHAR, stage_ord TINYINT, status_date VARCHAR)")
    for tid, centre, max_stage, events in rows:
        con.execute("INSERT INTO targets VALUES (?, ?)", [tid, centre])
        con.execute("INSERT INTO target_stage VALUES (?, ?, ?, 'xray')", [tid, centre, max_stage])
        for canon, ord_, date in events:
            con.execute("INSERT INTO status_history_canon VALUES (?, ?, ?, ?, ?)",
                        [tid, centre, canon, ord_, date])
    censoring.OUT = tmp_path
    return con


def cfg(**over):
    c = {
        "archive": {"freeze_date": FREEZE, "valid_from": "1995-01-01"},
        "censoring": {
            "centre_quantile": 0.995, "window_days": 180, "apply_freeze_window": True,
            "bulk_closure": {"enabled": True, "min_targets": 3, "min_pct_of_centre_stops": 5.0},
            "acceptance": {"min_pct": 0.0, "max_pct": 100.0},
        },
    }
    c["censoring"].update(over)
    return c


def result(con):
    return {r[0]: r for r in con.execute(
        "SELECT target_id, censored, censored_reason, censored_centre_winddown, "
        "censored_bulk_closure, reached_top_gate FROM censoring").fetchall()}


# --------------------------------------------------------------------------- the core rules

def test_deposited_target_is_never_censored(tmp_path):
    """Reaching the top gate is an answer, whatever else happened around it."""
    con = make_db(tmp_path, [
        ("A", "X", 8, [("selected", 0, "2010-01-01"), ("deposited", 8, "2010-06-30")]),
        # three stops on one date make it a bulk-closure date for centre X
        ("B", "X", 2, [("work_stopped", None, "2010-06-30")]),
        ("C", "X", 2, [("work_stopped", None, "2010-06-30")]),
        ("D", "X", 3, [("work_stopped", None, "2010-06-30")]),
    ])
    censoring.build(con, cfg())
    r = result(con)
    assert r["A"][1] is False and r["A"][5] is True
    assert all(r[t][1] is True for t in "BCD")


def test_bulk_closure_censors_a_mass_file_closure(tmp_path):
    """NESG stopped 34,605 targets on one day: the wind-down quantile cannot see it."""
    rows = [(f"T{i}", "NESG", 1, [("cloned", 1, "2005-01-01"), ("work_stopped", None, "2010-06-30")])
            for i in range(10)]
    # a long tail of later activity pushes the 99.5th percentile years past the closure
    rows += [(f"L{i}", "NESG", 4, [("purified", 4, "2015-02-25")]) for i in range(3)]
    con = make_db(tmp_path, rows)
    censoring.build(con, cfg())
    r = result(con)
    assert all(r[f"T{i}"][4] is True for i in range(10)), "mass closure not caught"
    assert all(r[f"T{i}"][3] is False for i in range(10)), "wind-down rule should miss this"
    assert r["T0"][2] == "bulk_closure"


def test_bulk_closure_respects_the_min_targets_floor(tmp_path):
    """Two targets stopping on one day is a coincidence, not an administrative act."""
    con = make_db(tmp_path, [
        ("A", "X", 1, [("work_stopped", None, "2008-05-05")]),
        ("B", "X", 1, [("work_stopped", None, "2008-05-05")]),
        ("C", "X", 4, [("purified", 4, "2016-01-01")]),
    ])
    censoring.build(con, cfg(bulk_closure={"enabled": True, "min_targets": 3,
                                           "min_pct_of_centre_stops": 5.0}))
    r = result(con)
    assert r["A"][4] is False and r["B"][4] is False


def test_bulk_closure_can_be_disabled(tmp_path):
    con = make_db(tmp_path, [
        (f"T{i}", "X", 1, [("work_stopped", None, "2010-06-30")]) for i in range(5)
    ] + [("L", "X", 4, [("purified", 4, "2016-01-01")])])
    censoring.build(con, cfg(bulk_closure={"enabled": False, "min_targets": 3,
                                           "min_pct_of_centre_stops": 5.0}))
    assert all(v[4] is False for v in result(con).values())


def test_centre_winddown_censors_targets_still_open_when_the_lights_went_out(tmp_path):
    con = make_db(tmp_path, [
        ("EARLY", "X", 2, [("expressed", 2, "2003-01-01")]),   # long dead before the centre stopped
        ("LATE", "X", 2, [("expressed", 2, "2011-12-01")]),    # still moving at the end
        ("LAST", "X", 3, [("soluble", 3, "2012-01-01")]),
    ])
    censoring.build(con, cfg())
    r = result(con)
    assert r["LATE"][3] is True and r["LAST"][3] is True
    assert r["EARLY"][3] is False, "a target dead nine years before the wind-down is a real failure"


def test_freeze_window_censors_targets_still_active_at_the_archive_freeze(tmp_path):
    con = make_db(tmp_path, [("A", "X", 3, [("soluble", 3, "2017-05-01")])])
    censoring.build(con, cfg())
    assert result(con)["A"][3] is True

    censoring.build(con, cfg(apply_freeze_window=False, centre_quantile=0.995, window_days=180,
                             bulk_closure={"enabled": True, "min_targets": 3, "min_pct_of_centre_stops": 5.0},
                             acceptance={"min_pct": 0.0, "max_pct": 100.0}))
    # with the freeze window off it is only caught by its own centre's wind-down
    assert result(con)["A"][3] is True  # single-target centre: its last date IS the wind-down


# --------------------------------------------------------------------------- date hygiene

def test_out_of_range_dates_are_dropped_not_clamped(tmp_path):
    """CSGID carries 5,136 events on 1979-01-01. Clamping would drag the centre's quantile back."""
    rows = [("BAD", "X", 1, [("cloned", 1, "1979-01-01")])]
    rows += [(f"OK{i}", "X", 2, [("expressed", 2, "2016-06-01")]) for i in range(4)]
    con = make_db(tmp_path, rows)
    censoring.build(con, cfg())
    last_any = dict(con.execute("SELECT target_id, last_any FROM censoring").fetchall())
    assert last_any["BAD"] is None, "a 1979 date must not become a real activity date"
    assert all(v is not None for k, v in last_any.items() if k.startswith("OK"))


def test_dates_after_the_freeze_are_dropped(tmp_path):
    con = make_db(tmp_path, [("A", "X", 1, [("cloned", 1, "2017-09-28")])])
    censoring.build(con, cfg())
    assert con.execute("SELECT last_any FROM censoring").fetchone()[0] is None


def test_target_with_no_usable_date_is_not_censored_by_winddown(tmp_path):
    con = make_db(tmp_path, [("A", "X", 1, [("cloned", 1, "1979-01-01")]),
                             ("B", "X", 4, [("purified", 4, "2016-01-01")])])
    censoring.build(con, cfg())
    assert result(con)["A"][3] is False, "no date means unknown, and unknown is not evidence"


# --------------------------------------------------------------------------- bookkeeping

def test_reason_column_agrees_with_the_flags(tmp_path):
    rows = [(f"T{i}", "X", 1, [("cloned", 1, "2010-06-01"), ("work_stopped", None, "2010-06-30")])
            for i in range(5)]
    con = make_db(tmp_path, rows)
    censoring.build(con, cfg())
    for _, cens, reason, a, b, _ in result(con).values():
        assert cens == (a or b)
        expect = "both" if a and b else "bulk_closure" if b else "centre_winddown" if a else ""
        assert reason == expect


def test_one_row_per_target(tmp_path):
    rows = [("A", "X", 1, [("cloned", 1, "2010-01-01"), ("work_stopped", None, "2010-06-30")]),
            ("B", "Y", 2, [("expressed", 2, "2011-01-01")])]
    con = make_db(tmp_path, rows)
    censoring.build(con, cfg())
    assert con.execute("SELECT count(*), count(DISTINCT target_id) FROM censoring").fetchone() == (2, 2)


@pytest.mark.skipif(not (ROOT / "data" / "parquet" / "censoring" / "censoring.parquet").exists(),
                    reason="needs the built archive")
def test_real_archive_is_inside_the_acceptance_gate():
    con = duckdb.connect(str(ROOT / "data" / "faffabout.duckdb"), read_only=True)
    pct = con.execute("SELECT round(100.0*avg(CASE WHEN censored THEN 1 ELSE 0 END), 2) FROM censoring").fetchone()[0]
    assert 12.0 <= pct <= 22.0, f"censoring rate {pct}% is outside the 12 to 22% gate"


@pytest.mark.skipif(not (ROOT / "data" / "parquet" / "censoring" / "censoring.parquet").exists(),
                    reason="needs the built archive")
def test_no_deposited_target_is_ever_censored_in_the_real_archive():
    con = duckdb.connect(str(ROOT / "data" / "faffabout.duckdb"), read_only=True)
    assert con.execute("SELECT count(*) FROM censoring WHERE censored AND reached_top_gate").fetchone()[0] == 0
