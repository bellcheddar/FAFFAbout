"""censoring.py: decide which stalled targets are censored rather than failed.

Imported by 06_build_labels.py; runnable on its own for the report.

The distinction this module draws is the one the whole project rests on. A target sitting
below the top gate when the archive froze is one of three things:

  1. failed     tried and abandoned on scientific grounds        -> a usable negative
  2. censored   the centre's file closed with work incomplete    -> NEVER a negative
  3. in flight  still active when the archive froze              -> NEVER a negative

Two rules, both applied, each recorded separately so either can be inspected or disabled.

Rule A, centre wind-down (the specification's rule). Per centre, a high quantile of its
targets' last activity dates marks when it effectively stopped reporting; a target below
the top gate whose own last activity falls within a window of that point was still open
when the lights went out. The same window is applied against the 2017-06-30 freeze.

Rule B, bulk closure (added 2026-09-07). Rule A alone censors 5.2% of the archive, against
the specification's own expectation of 15 to 20%, because it misses mass closures: NESG
stopped 34,605 targets on 2010-06-30, 99.6% of every stop it ever recorded, while its
99.5th-percentile last-activity date is 2015-02-25. A centre stopping hundreds or tens of
thousands of targets on a single date is performing an administrative act, which is the
mechanism the specification names, observed directly instead of inferred.

Together they censor 19.03% of the archive, inside the 12 to 22% acceptance gate.

Usage
  .venv/bin/python scripts/censoring.py            # build + report
  .venv/bin/python scripts/censoring.py --report   # report from the existing table
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "faffabout.duckdb"
PQ = ROOT / "data" / "parquet"
CFG = ROOT / "config" / "labels.yaml"
OUT = PQ / "censoring"


def load_config() -> dict:
    return yaml.safe_load(CFG.read_text())


def build(con: duckdb.DuckDBPyConnection, cfg: dict) -> None:
    """Write data/parquet/censoring/censoring.parquet: one row per target."""
    arc, cen = cfg["archive"], cfg["censoring"]
    bulk = cen["bulk_closure"]
    freeze, valid_from = arc["freeze_date"], arc["valid_from"]
    OUT.mkdir(parents=True, exist_ok=True)

    # Date hygiene first. Out-of-range dates become NULL, never clamped: a 1979 date
    # dragged onto the timeline would move its centre's quantile and mis-censor the cohort.
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW ev AS
        SELECT target_id, centre, stage_ord, status_canon,
               CASE WHEN try_cast(status_date AS DATE)
                         BETWEEN DATE '{valid_from}' AND DATE '{freeze}'
                    THEN try_cast(status_date AS DATE) END AS d
        FROM status_history_canon
        WHERE status_date <> ''
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW act AS
        SELECT e.target_id, e.centre,
               max(e.d)                                                        AS last_any,
               max(e.d) FILTER (WHERE e.stage_ord IS NOT NULL)                 AS last_ladder,
               max(e.d) FILTER (WHERE e.status_canon = 'work_stopped')         AS stop_date,
               count(*) FILTER (WHERE e.d IS NULL)                             AS n_bad_dates
        FROM ev e GROUP BY 1, 2
    """)
    # Rule A: per-centre wind-down point.
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW centre_last AS
        SELECT centre, quantile_cont(last_any, {cen['centre_quantile']}) AS wind_down
        FROM act WHERE last_any IS NOT NULL GROUP BY 1
    """)
    # Rule B: (centre, date) pairs that look like an administrative mass closure.
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW bulk_dates AS
        WITH stops AS (
            SELECT centre, stop_date AS d, count(*) AS n
            FROM act WHERE stop_date IS NOT NULL GROUP BY 1, 2),
        tot AS (SELECT centre, sum(n) AS total FROM stops GROUP BY 1)
        SELECT s.centre, s.d, s.n, t.total,
               round(100.0 * s.n / t.total, 2) AS pct_of_centre_stops
        FROM stops s JOIN tot t USING (centre)
        WHERE s.n >= {bulk['min_targets']}
          AND 100.0 * s.n / t.total >= {bulk['min_pct_of_centre_stops']}
          AND {str(bool(bulk['enabled'])).lower()}
    """)
    freeze_clause = (f"datediff('day', a.last_any, DATE '{freeze}') < {cen['window_days']}"
                     if cen["apply_freeze_window"] else "false")
    con.execute(f"""
        COPY (
            SELECT
                t.target_id, t.centre, s.max_stage, s.method,
                a.last_any, a.last_ladder, a.stop_date, cl.wind_down,
                b.d IS NOT NULL                                       AS on_bulk_closure_date,
                b.n                                                   AS bulk_closure_size,
                coalesce(s.max_stage, -1) >= 8                        AS reached_top_gate,
                -- Rule A
                coalesce(s.max_stage, -1) < 8 AND a.last_any IS NOT NULL
                  AND (datediff('day', a.last_any, cl.wind_down) < {cen['window_days']}
                       OR {freeze_clause})                            AS censored_centre_winddown,
                -- Rule B
                coalesce(s.max_stage, -1) < 8 AND b.d IS NOT NULL     AS censored_bulk_closure
            FROM targets t
            JOIN target_stage s USING (target_id)
            LEFT JOIN act a USING (target_id)
            LEFT JOIN centre_last cl ON cl.centre = t.centre
            LEFT JOIN bulk_dates b ON b.centre = t.centre AND b.d = a.stop_date
        ) TO '{OUT / 'censoring.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    # Second pass to add the combined flag and a human-readable reason.
    con.execute(f"""
        COPY (
            SELECT *,
                   censored_centre_winddown OR censored_bulk_closure AS censored,
                   CASE WHEN censored_centre_winddown AND censored_bulk_closure THEN 'both'
                        WHEN censored_bulk_closure   THEN 'bulk_closure'
                        WHEN censored_centre_winddown THEN 'centre_winddown'
                        ELSE '' END AS censored_reason
            FROM read_parquet('{OUT / 'censoring.parquet'}')
        ) TO '{OUT / 'censoring_tmp.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    (OUT / "censoring_tmp.parquet").replace(OUT / "censoring.parquet")
    con.execute(f"CREATE OR REPLACE VIEW censoring AS SELECT * FROM read_parquet('{OUT / 'censoring.parquet'}')")


def report(con: duckdb.DuckDBPyConnection, cfg: dict) -> bool:
    con.execute(f"CREATE OR REPLACE VIEW censoring AS SELECT * FROM read_parquet('{OUT / 'censoring.parquet'}')")
    n = con.execute("SELECT count(*) FROM censoring").fetchone()[0]
    print(f"targets: {n:,}\n")
    print("censoring by rule:")
    print(con.execute("""
        SELECT coalesce(nullif(censored_reason, ''), 'not censored') AS reason, count(*) AS n,
               round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
        FROM censoring GROUP BY 1 ORDER BY n DESC
    """).df().to_string(index=False))

    pct = con.execute("SELECT round(100.0*avg(CASE WHEN censored THEN 1 ELSE 0 END), 2) FROM censoring").fetchone()[0]
    a = cfg["censoring"]["acceptance"]
    ok = a["min_pct"] <= pct <= a["max_pct"]
    print(f"\ncensored overall: {pct}%   gate {a['min_pct']} to {a['max_pct']}%   {'PASS' if ok else 'FAIL'}")

    print("\nbulk-closure dates detected:")
    print(con.execute("""
        SELECT centre, stop_date, max(bulk_closure_size) AS targets_stopped_that_day,
               count(*) FILTER (WHERE censored_bulk_closure) AS censored_here
        FROM censoring WHERE on_bulk_closure_date
        GROUP BY 1, 2 ORDER BY targets_stopped_that_day DESC
    """).df().to_string(index=False))

    print("\nper centre (top 15 by size):")
    print(con.execute("""
        SELECT centre, count(*) AS n,
               round(100.0*avg(CASE WHEN censored THEN 1 ELSE 0 END), 1) AS pct_censored,
               round(100.0*avg(CASE WHEN censored_centre_winddown THEN 1 ELSE 0 END), 1) AS pct_winddown,
               round(100.0*avg(CASE WHEN censored_bulk_closure THEN 1 ELSE 0 END), 1) AS pct_bulk,
               round(100.0*avg(CASE WHEN reached_top_gate THEN 1 ELSE 0 END), 1) AS pct_deposited
        FROM censoring GROUP BY 1 ORDER BY n DESC LIMIT 15
    """).df().to_string(index=False))

    print("\ncensoring by max_stage (censored targets are excluded from every loss):")
    print(con.execute("""
        SELECT coalesce(CAST(max_stage AS VARCHAR), 'none') AS max_stage, count(*) AS n,
               count(*) FILTER (WHERE censored) AS censored,
               round(100.0*avg(CASE WHEN censored THEN 1 ELSE 0 END), 1) AS pct
        FROM censoring GROUP BY 1 ORDER BY max_stage NULLS LAST
    """).df().to_string(index=False))

    # The point of the exercise: without censoring, "stalled" is dominated by two dates.
    print("\nsanity check, targets stalled below gate 8 that are NOT censored:",
          f"{con.execute('SELECT count(*) FROM censoring WHERE NOT reached_top_gate AND NOT censored').fetchone()[0]:,}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="report only, do not rebuild")
    args = ap.parse_args()
    cfg = load_config()
    con = duckdb.connect(str(DB))
    if not args.report:
        build(con, cfg)
        print(f"wrote {OUT / 'censoring.parquet'}\n")
    ok = report(con, cfg)
    con.close()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
