#!/usr/bin/env python
"""build_duckdb.py: attach DuckDB over the Parquet directory and run the Phase 1 acceptance test.

Creates data/faffabout.duckdb with one VIEW per Parquet table (the Parquet files stay the
source of truth; the .duckdb file is a convenience handle and is read-only at runtime) plus
a few derived views the later phases and the app query directly.

Views
  targets, target_sequences, status_history, status_history_canon, trials, protocols,
  outcomes, status_map, target_stage, clusters
  target_summary   targets x target_stage x clusters x outcomes, one row per target

Usage
  .venv/bin/python scripts/build_duckdb.py            # build + acceptance report
  .venv/bin/python scripts/build_duckdb.py --check    # acceptance report only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
PQ = ROOT / "data" / "parquet"
CL = ROOT / "data" / "clusters"
DB = ROOT / "data" / "faffabout.duckdb"

PARQUET_VIEWS = {
    "targets": PQ / "targets" / "*.parquet",
    "target_sequences": PQ / "target_sequences" / "*.parquet",
    "status_history": PQ / "status_history" / "*.parquet",
    "status_history_canon": PQ / "status_history_canon" / "*.parquet",
    "trials": PQ / "trials" / "*.parquet",
    "protocols": PQ / "protocols" / "*.parquet",
    "outcomes": PQ / "outcomes" / "*.parquet",
    "status_map": PQ / "status_map.parquet",
    "target_stage": PQ / "target_stage" / "*.parquet",
    "clusters": CL / "tt30_clusters.parquet",
}


def build(con: duckdb.DuckDBPyConnection) -> None:
    for name, glob in PARQUET_VIEWS.items():
        if not list(glob.parent.glob(glob.name)):
            print(f"  (skipping view {name}: no files at {glob})")
            continue
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{glob}')")
    con.execute("""
        CREATE OR REPLACE VIEW target_summary AS
        SELECT t.target_id, t.centre, t.local_id, t.organism, t.taxon_id, t.seq_len, t.seq_md5,
               t.chem_type, t.construct_type, t.protein_types, t.categories,
               t.status_raw AS target_status_raw, t.stop_status, t.n_trials,
               t.first_seen, t.last_seen,
               s.max_stage, s.method, s.n_events, s.n_ladder_events, s.first_event, s.last_event, s.stopped,
               c.cluster_id, c.cluster_size,
               o.n_pdb, o.first_deposit
        FROM targets t
        LEFT JOIN target_stage s USING (target_id)
        LEFT JOIN clusters c USING (seq_md5)
        LEFT JOIN (SELECT target_id, count(DISTINCT pdb_id) n_pdb, min(NULLIF(deposit_date, '')) first_deposit
                   FROM outcomes GROUP BY 1) o USING (target_id)
    """)


def acceptance(con: duckdb.DuckDBPyConnection) -> bool:
    ok = True
    n = con.execute("SELECT count(*) FROM targets").fetchone()[0]
    print(f"targets                        {n:>10,}   (expect low 300,000s)")
    ok &= 300_000 <= n <= 400_000

    dup = con.execute("SELECT count(*) - count(DISTINCT target_id) FROM targets").fetchone()[0]
    print(f"duplicate target_id            {dup:>10,}")
    ok &= dup == 0

    for t in ("status_history", "trials", "protocols", "outcomes", "target_sequences"):
        print(f"{t:<30} {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:>10,}")

    nulls = con.execute("SELECT count(*) FROM status_history_canon WHERE status_canon IS NULL").fetchone()[0]
    unm = con.execute("SELECT count(DISTINCT status_raw) FROM status_history_canon WHERE status_canon IS NULL").fetchone()[0]
    print(f"status_canon NULL rows         {nulls:>10,}   over {unm} distinct raw values (must all be in status_unmapped.csv)")
    unmapped_csv = PQ / "status_unmapped.csv"
    documented = 0
    if unmapped_csv.exists():
        documented = max(0, sum(1 for _ in unmapped_csv.open()) - 1)
    print(f"documented unmapped values     {documented:>10,}")
    ok &= unm == documented

    prot = con.execute("SELECT count(*) FROM targets WHERE chem_type = 'protein' AND seq_len > 0").fetchone()[0]
    print(f"protein targets with sequence  {prot:>10,}")
    has_cluster = con.execute("""
        SELECT count(*) FROM targets t LEFT JOIN clusters c USING (seq_md5)
        WHERE t.chem_type = 'protein' AND t.seq_len > 0 AND c.cluster_id IS NULL
    """).fetchone()[0]
    print(f"protein targets w/o cluster    {has_cluster:>10,}   (must be 0)")
    ok &= has_cluster == 0
    ncl = con.execute("SELECT count(DISTINCT cluster_id) FROM clusters").fetchone()[0]
    print(f"clusters (30% id)              {ncl:>10,}")

    print("\ntargets per centre (top 12):")
    print(con.execute("""
        SELECT centre, count(*) n, round(avg(seq_len)) mean_len,
               round(100.0 * avg(CASE WHEN max_stage >= 8 THEN 1 ELSE 0 END), 1) pct_deposited,
               round(100.0 * avg(CASE WHEN stopped THEN 1 ELSE 0 END), 1) pct_stopped
        FROM target_summary GROUP BY 1 ORDER BY n DESC LIMIT 12
    """).df().to_string(index=False))
    print("\nmax_stage distribution:")
    print(con.execute("""
        SELECT coalesce(CAST(max_stage AS VARCHAR), 'none') max_stage, count(*) n,
               round(100.0 * count(*) / sum(count(*)) OVER (), 2) pct
        FROM target_summary GROUP BY 1 ORDER BY max_stage NULLS LAST
    """).df().to_string(index=False))
    print("\nPHASE 1 ACCEPTANCE:", "PASS" if ok else "FAIL")
    return bool(ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    con = duckdb.connect(str(DB))
    if not args.check:
        build(con)
        print(f"views built in {DB}")
    ok = acceptance(con)
    con.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
