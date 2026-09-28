#!/usr/bin/env python
"""build_serving_db.py: a self-contained DuckDB for the web app, with no Parquet behind it.

data/faffabout.duckdb is a query layer: nearly every name in it is a VIEW over
read_parquet('/Users/dellboy/.../data/parquet/...'), an absolute path on the build machine.
Shipped to the droplet on its own (as deploy.sh did until 2026-09-28), every view pointed
at a path that does not exist there: /healthz still said 200, because it only checked that
files existed, and archive_stats swallowed the error and reported 0 targets.

This materialises exactly what app/ queries, as real tables with the same names, so the
serving queries run unchanged:

  targets               target_id, centre, organism, seq_md5, stop_status
  censoring             all columns (per-target outcome, censoring and method)
  features              target_id, host, tag
  outcomes              target_id, pdb_id, deposit_date
  trials                the six columns app/relatives.py reads
  status_history_canon  ONE row per trial with its furthest stage_ord (all the panel uses)
  taxonomy_by_id, taxonomy_lookup

deploy.sh ships the result as data/faffabout.duckdb on the droplet.

Usage
  .venv/bin/python scripts/build_serving_db.py
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "faffabout.duckdb"
OUT = ROOT / "data" / "faffabout_serving.duckdb"

TABLES = {
    "targets": "SELECT target_id, centre, organism, seq_md5, stop_status FROM src.targets",
    "censoring": "SELECT * FROM src.censoring",
    "features": "SELECT target_id, host, tag FROM src.features",
    "outcomes": "SELECT target_id, pdb_id, deposit_date FROM src.outcomes",
    "trials": """SELECT target_id, trial_id, construct_type, stop_status, stop_remark, free_text_notes
                 FROM src.trials""",
    "status_history_canon": """SELECT target_id, trial_id, max(stage_ord) AS stage_ord
                               FROM src.status_history_canon
                               WHERE trial_id IS NOT NULL GROUP BY ALL""",
    "taxonomy_by_id": "SELECT * FROM src.taxonomy_by_id",
    "taxonomy_lookup": "SELECT * FROM src.taxonomy_lookup",
}


def main() -> None:
    t0 = time.time()
    tmp = OUT.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    con.execute(f"ATTACH '{SRC}' AS src (READ_ONLY)")
    for name, sql in TABLES.items():
        con.execute(f"CREATE TABLE {name} AS {sql}")
        n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        print(f"  {name:<22} {n:>10,} rows")
    # Every name the app queries must be a real table, never a view over a local path.
    views = con.execute("SELECT count(*) FROM duckdb_views() WHERE NOT internal AND database_name = current_database()").fetchone()[0]
    assert views == 0, f"{views} views in the serving database"
    con.execute("DETACH src")
    con.execute("CHECKPOINT")
    con.close()
    tmp.replace(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
