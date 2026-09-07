"""splits.py: the three evaluation splits, all reported, never mixed.

Imported by 07_build_sft.py; runnable on its own for the report.

1. Cluster-held-out (primary). 80/10/10 by MMseqs2 30%-identity cluster, never by row.
   The archive attempts the same protein across five or more orthologues at different
   centres, so a random row split leaks and returns a meaningless 0.9 AUC. Clusters are
   assigned largest-first to whichever split is furthest below its quota, which keeps
   whole clusters intact while landing within a fraction of a percent of 80/10/10 by
   target count. A hash-based assignment would leave the split sizes at the mercy of a
   447-member cluster.

2. Leave-one-centre-out. Derived at evaluation time from the `centre` column across
   JCSG, NESG, MCSG, NYSGRC and CESG, rather than materialised five times over. `centre`
   is also an input feature, so the model conditions on pipeline differences openly
   instead of absorbing them.

3. Temporal (the honest forecasting test). Train on targets whose work finished before
   2014; test on targets whose work started in 2014 or later. Targets straddling the
   boundary belong to neither: a target selected in 2013 and deposited in 2016 would
   otherwise carry post-boundary information into training.

Usage
  .venv/bin/python scripts/splits.py            # build + report
  .venv/bin/python scripts/splits.py --report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "faffabout.duckdb"
PQ = ROOT / "data" / "parquet"
CL = ROOT / "data" / "clusters"
OUT = PQ / "splits"

RATIOS = {"train": 0.80, "valid": 0.10, "test": 0.10}
TEMPORAL_BOUNDARY = "2014-01-01"
LOCO_CENTRES = ["JCSG", "NESG", "MCSG", "NYSGRC", "CESG"]


def assign_clusters(sizes: list[tuple[int, int]]) -> dict[int, str]:
    """Largest cluster first into whichever split is furthest below its quota.

    Deterministic, keeps every cluster whole, and balances target counts far better than
    hashing: one 447-member cluster landing in `valid` would move it by 0.16 points.
    """
    total = sum(n for _, n in sizes)
    quota = {k: total * r for k, r in RATIOS.items()}
    have = dict.fromkeys(RATIOS, 0)
    out: dict[int, str] = {}
    for cid, n in sorted(sizes, key=lambda kv: (-kv[1], kv[0])):
        pick = max(RATIOS, key=lambda k: (quota[k] - have[k], k))
        out[cid] = pick
        have[pick] += n
    return out


def build(con: duckdb.DuckDBPyConnection) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con.execute(f"CREATE OR REPLACE VIEW clusters_30 AS SELECT * FROM read_parquet('{CL / 'tt30_clusters.parquet'}')")
    con.execute(f"CREATE OR REPLACE VIEW censoring AS SELECT * FROM read_parquet('{PQ}/censoring/censoring.parquet')")

    # Per target: its cluster, its centre, and its clean activity span.
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW base AS
        SELECT c.target_id, c.centre, c.max_stage, c.censored,
               cl.cluster_id, t.seq_md5,
               c.last_any AS last_event,
               (SELECT min(CASE WHEN try_cast(h.status_date AS DATE)
                                     BETWEEN DATE '1995-01-01' AND DATE '2017-06-30'
                                THEN try_cast(h.status_date AS DATE) END)
                FROM status_history_canon h WHERE h.target_id = c.target_id) AS first_event
        FROM censoring c
        JOIN targets t USING (target_id)
        LEFT JOIN clusters_30 cl ON cl.seq_md5 = t.seq_md5
    """)

    sizes = con.execute("""
        SELECT cluster_id, count(*) FROM base WHERE cluster_id IS NOT NULL GROUP BY 1
    """).fetchall()
    assignment = assign_clusters([(int(c), int(n)) for c, n in sizes])

    con.execute("CREATE OR REPLACE TEMP TABLE cl_split (cluster_id INTEGER, split VARCHAR)")
    con.executemany("INSERT INTO cl_split VALUES (?, ?)", list(assignment.items()))

    con.execute(f"""
        COPY (
            SELECT b.target_id, b.centre, b.cluster_id, b.max_stage, b.censored,
                   b.first_event, b.last_event,
                   coalesce(s.split, 'none')                            AS split_cluster,
                   CASE WHEN b.last_event  <  DATE '{TEMPORAL_BOUNDARY}' THEN 'train'
                        WHEN b.first_event >= DATE '{TEMPORAL_BOUNDARY}' THEN 'test'
                        ELSE 'straddles' END                            AS split_temporal,
                   b.centre IN ({', '.join(f"'{c}'" for c in LOCO_CENTRES)}) AS loco_eligible
            FROM base b LEFT JOIN cl_split s USING (cluster_id)
        ) TO '{OUT / 'splits.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
    """)
    con.execute(f"CREATE OR REPLACE VIEW splits AS SELECT * FROM read_parquet('{OUT / 'splits.parquet'}')")


def report(con: duckdb.DuckDBPyConnection) -> bool:
    con.execute(f"CREATE OR REPLACE VIEW splits AS SELECT * FROM read_parquet('{OUT / 'splits.parquet'}')")
    ok = True
    print("=== 1. cluster-held-out (primary) ===")
    print(con.execute("""
        SELECT split_cluster, count(*) AS targets,
               round(100.0*count(*)/sum(count(*)) OVER (), 2) AS pct,
               count(DISTINCT cluster_id) AS clusters,
               count(*) FILTER (WHERE NOT censored) AS uncensored,
               round(100.0*avg(CASE WHEN max_stage >= 8 THEN 1 ELSE 0 END), 2) AS pct_deposited
        FROM splits GROUP BY 1 ORDER BY targets DESC
    """).df().to_string(index=False))

    leak = con.execute("""
        SELECT count(*) FROM (
            SELECT cluster_id FROM splits WHERE cluster_id IS NOT NULL
            GROUP BY 1 HAVING count(DISTINCT split_cluster) > 1)
    """).fetchone()[0]
    print(f"\nclusters appearing in more than one split (must be 0): {leak}")
    ok &= leak == 0

    # A sequence in two splits would leak just as badly as a cluster in two splits.
    md5_leak = con.execute("""
        SELECT count(*) FROM (
            SELECT s.seq_md5 FROM (
                SELECT t.seq_md5, sp.split_cluster FROM splits sp JOIN targets t USING (target_id)
                WHERE t.seq_md5 <> '') s
            GROUP BY 1 HAVING count(DISTINCT s.split_cluster) > 1)
    """).fetchone()[0]
    print(f"identical sequences spanning splits (must be 0): {md5_leak}")
    ok &= md5_leak == 0

    for k, want in RATIOS.items():
        got = con.execute(f"""
            SELECT 1.0*count(*) FILTER (WHERE split_cluster = '{k}') / count(*)
            FROM splits WHERE split_cluster <> 'none'""").fetchone()[0]
        drift = abs(got - want) * 100
        print(f"  {k:<6} {got*100:6.2f}%  target {want*100:.0f}%  drift {drift:.2f} pts")
        ok &= drift < 1.0

    print("\n=== 2. leave-one-centre-out ===")
    print(con.execute(f"""
        SELECT centre, count(*) AS targets,
               count(*) FILTER (WHERE NOT censored) AS uncensored,
               round(100.0*avg(CASE WHEN max_stage >= 8 THEN 1 ELSE 0 END), 2) AS pct_deposited,
               round(100.0*avg(CASE WHEN censored THEN 1 ELSE 0 END), 1) AS pct_censored
        FROM splits WHERE centre IN ({', '.join(f"'{c}'" for c in LOCO_CENTRES)})
        GROUP BY 1 ORDER BY targets DESC
    """).df().to_string(index=False))

    print("\n=== 3. temporal (train pre-2014, test 2014+) ===")
    print(con.execute("""
        SELECT split_temporal, count(*) AS targets,
               round(100.0*count(*)/sum(count(*)) OVER (), 2) AS pct,
               count(*) FILTER (WHERE NOT censored) AS uncensored,
               round(100.0*avg(CASE WHEN max_stage >= 8 THEN 1 ELSE 0 END), 2) AS pct_deposited,
               min(first_event) AS earliest, max(last_event) AS latest
        FROM splits GROUP BY 1 ORDER BY targets DESC
    """).df().to_string(index=False))
    bad = con.execute(f"""
        SELECT count(*) FROM splits
        WHERE split_temporal = 'train' AND last_event >= DATE '{TEMPORAL_BOUNDARY}'
    """).fetchone()[0]
    print(f"\ntemporal train rows with activity after the boundary (must be 0): {bad}")
    ok &= bad == 0
    n_test = con.execute("SELECT count(*) FROM splits WHERE split_temporal = 'test'").fetchone()[0]
    print(f"temporal test set size: {n_test:,}  {'PASS' if n_test >= 5000 else 'FAIL (too small to evaluate)'}")
    ok &= n_test >= 5000
    return bool(ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    con = duckdb.connect(str(DB))
    if not args.report:
        build(con)
        print(f"wrote {OUT / 'splits.parquet'}\n")
    ok = report(con)
    print("\nSPLITS:", "PASS" if ok else "FAIL")
    con.close()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
