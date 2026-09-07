#!/usr/bin/env python
"""06_build_labels.py: the L1, L2 and L3 label sets.

Depends on scripts/censoring.py having run (data/parquet/censoring/).

L1, gate transitions. The workhorse. For every target and every gate it ACTUALLY ENTERED,
one row labelled `cleared`, `failed` or `censored`. Rows for gates a target never reached
are not emitted at all: not-attempted is not a negative, and a target that stopped at
`selected` says nothing about crystallisation.

    target with max_stage 4, uncensored ->  gates 0,1,2,3 cleared,  gate 4 failed
    target with max_stage 4, censored   ->  gates 0,1,2,3 cleared,  gate 4 censored
    target with max_stage 8             ->  gates 0..8 all cleared

A censored target still yields valid `cleared` rows for every gate it passed. Only its
terminal gate is unknown, and that row is excluded from the loss while staying available
as retrieval context.

L2, terminal outcome. One row per uncensored target: `deposited` or `stalled_at_{g}`.
For Brier score and expected calibration error, not for the SFT loss.

L3, within-cluster preference pairs. Within an MMseqs2 cluster, uncensored pairs where
max_stage(A) > max_stage(B) + 1: two near-identical proteins, different construct or host,
different fate. Capped per cluster so large families cannot dominate. Pairs whose members
also share a 70%-identity cluster are marked `hard`, which is the stratum worth roughly
fifty random negatives.

Usage
  .venv/bin/python scripts/06_build_labels.py
  .venv/bin/python scripts/06_build_labels.py --report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "faffabout.duckdb"
PQ = ROOT / "data" / "parquet"
CL = ROOT / "data" / "clusters"
CFG = ROOT / "config" / "labels.yaml"

LADDER = ["selected", "cloned", "expressed", "soluble", "purified",
          "crystallised", "diffracting", "structure", "deposited"]
# A gate is the transition OUT of a stage, so gate g asks "did this target get past
# stage g". There are eight gates on the nine-rung ladder: gate 7 (structure ->
# deposited) is the last one, and stage 8 is terminal.
N_GATES = 8


def attach_views(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"CREATE OR REPLACE VIEW censoring AS SELECT * FROM read_parquet('{PQ}/censoring/censoring.parquet')")
    for tag in ("tt30", "tt70"):
        f = CL / f"{tag}_clusters.parquet"
        if f.exists():
            con.execute(f"CREATE OR REPLACE VIEW clusters_{tag[2:]} AS SELECT * FROM read_parquet('{f}')")


def build_l1(con: duckdb.DuckDBPyConnection) -> None:
    """One row per (target, gate entered)."""
    out = PQ / "labels_l1"
    out.mkdir(parents=True, exist_ok=True)
    gate_name_case = ("CASE g.gate " + " ".join(f"WHEN {i} THEN '{n}'" for i, n in enumerate(LADDER)) + " END")
    con.execute(f"""
        COPY (
            WITH t AS (
                SELECT c.target_id, c.centre, c.max_stage, c.method, c.censored, c.censored_reason,
                       tg.seq_md5, tg.organism, tg.seq_len
                FROM censoring c JOIN targets tg USING (target_id)
                WHERE c.max_stage IS NOT NULL
            ),
            -- Gates 0 to 7 only. A gate is a TRANSITION out of stage g, so there are
            -- eight of them on a nine-rung ladder: nothing lies beyond deposition.
            -- Emitting a gate-8 row applies the spec's rule literally (cleared iff
            -- max_stage > 8, impossible) and labels all 10,500 deposited targets
            -- `failed` at the final gate, teaching the model that deposition always fails.
            gates AS (SELECT unnest(generate_series(0, 7)) AS gate)
            SELECT t.target_id, t.centre, t.method, t.seq_md5, t.organism, t.seq_len,
                   g.gate,
                   {gate_name_case} AS gate_name,
                   t.max_stage,
                   CASE WHEN t.max_stage > g.gate THEN 'cleared'
                        WHEN t.censored           THEN 'censored'
                        ELSE 'failed' END AS label,
                   -- censored rows are excluded from the loss but kept for retrieval
                   (t.max_stage > g.gate OR NOT t.censored) AS in_loss,
                   t.censored, t.censored_reason
            FROM t JOIN gates g ON g.gate <= least(t.max_stage, 7)
        ) TO '{out / 'l1.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
    """)
    con.execute(f"CREATE OR REPLACE VIEW labels_l1 AS SELECT * FROM read_parquet('{out / 'l1.parquet'}')")


def build_l2(con: duckdb.DuckDBPyConnection) -> None:
    """One row per uncensored target: the terminal outcome."""
    out = PQ / "labels_l2"
    out.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT c.target_id, c.centre, c.method, tg.seq_md5, tg.organism, tg.seq_len,
                   c.max_stage,
                   CASE WHEN c.max_stage >= 8 THEN 'deposited'
                        ELSE 'stalled_at_' || CAST(c.max_stage AS VARCHAR) END AS outcome,
                   c.max_stage >= 8 AS deposited
            FROM censoring c JOIN targets tg USING (target_id)
            WHERE NOT c.censored AND c.max_stage IS NOT NULL
        ) TO '{out / 'l2.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
    """)
    con.execute(f"CREATE OR REPLACE VIEW labels_l2 AS SELECT * FROM read_parquet('{out / 'l2.parquet'}')")


def build_l3(con: duckdb.DuckDBPyConnection, cap: int) -> None:
    """Within-cluster preference pairs, capped per cluster."""
    out = PQ / "labels_l3"
    out.mkdir(parents=True, exist_ok=True)
    # Match tables as well as views: looking only at duckdb_views() made the hard-negative
    # flag silently all-false whenever clusters_70 happened to be a table, and the hard
    # stratum is the one the spec values at roughly fifty random negatives.
    has70 = con.execute("""
        SELECT count(*) FROM information_schema.tables WHERE table_name = 'clusters_70'
    """).fetchone()[0] > 0
    if not has70:
        print("  WARNING: no clusters_70 relation, so no pair can be marked hard.\n"
              "           Run: bash scripts/04_cluster_sequences.sh --min-seq-id 0.70")
    hard_expr = ("a.c70 IS NOT NULL AND a.c70 = b.c70" if has70 else "false")
    con.execute(f"""
        COPY (
            WITH m AS (
                SELECT c.target_id, c.centre, c.method, c.max_stage, tg.seq_md5, tg.organism,
                       tg.seq_len, cl.cluster_id AS c30
                       {', c7.cluster_id AS c70' if has70 else ', NULL AS c70'}
                FROM censoring c
                JOIN targets tg USING (target_id)
                JOIN clusters_30 cl ON cl.seq_md5 = tg.seq_md5
                {'LEFT JOIN clusters_70 c7 ON c7.seq_md5 = tg.seq_md5' if has70 else ''}
                WHERE NOT c.censored AND c.max_stage IS NOT NULL
            ),
            pairs AS (
                SELECT a.c30 AS cluster_id,
                       a.target_id AS chosen_id,   a.centre AS chosen_centre,
                       a.max_stage AS chosen_stage, a.organism AS chosen_organism,
                       a.seq_len AS chosen_len,    a.method AS chosen_method,
                       b.target_id AS rejected_id, b.centre AS rejected_centre,
                       b.max_stage AS rejected_stage, b.organism AS rejected_organism,
                       b.seq_len AS rejected_len,  b.method AS rejected_method,
                       a.max_stage - b.max_stage AS stage_gap,
                       {hard_expr} AS hard,
                       a.centre <> b.centre AS cross_centre,
                       row_number() OVER (
                           PARTITION BY a.c30
                           -- hard pairs first, then the widest fate gap: the cap keeps
                           -- the most informative pairs rather than an arbitrary slice
                           ORDER BY ({hard_expr}) DESC, a.max_stage - b.max_stage DESC, a.target_id
                       ) AS rn
                FROM m a JOIN m b
                  ON a.c30 = b.c30 AND a.max_stage > b.max_stage + 1
            )
            SELECT * EXCLUDE (rn) FROM pairs WHERE rn <= {cap}
        ) TO '{out / 'l3.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
    """)
    con.execute(f"CREATE OR REPLACE VIEW labels_l3 AS SELECT * FROM read_parquet('{out / 'l3.parquet'}')")


def report(con: duckdb.DuckDBPyConnection) -> bool:
    ok = True
    print("=== L1: gate transitions ===")
    print(con.execute("""
        SELECT gate, any_value(gate_name) AS gate_name, count(*) AS rows,
               count(*) FILTER (WHERE label = 'cleared')  AS cleared,
               count(*) FILTER (WHERE label = 'failed')   AS failed,
               count(*) FILTER (WHERE label = 'censored') AS censored,
               round(100.0 * count(*) FILTER (WHERE label = 'cleared')
                     / nullif(count(*) FILTER (WHERE in_loss), 0), 1) AS pct_cleared_in_loss
        FROM labels_l1 GROUP BY gate ORDER BY gate
    """).df().to_string(index=False))
    n1, in_loss = con.execute("SELECT count(*), count(*) FILTER (WHERE in_loss) FROM labels_l1").fetchone()
    print(f"\ntotal L1 rows: {n1:,}   in loss: {in_loss:,}   held out as censored: {n1 - in_loss:,}")

    bad = con.execute("SELECT count(*) FROM labels_l1 WHERE gate > max_stage").fetchone()[0]
    print(f"rows for gates never entered (must be 0): {bad}")
    ok &= bad == 0
    bad2 = con.execute("SELECT count(*) FROM labels_l1 WHERE label = 'censored' AND in_loss").fetchone()[0]
    print(f"censored rows leaking into the loss (must be 0): {bad2}")
    ok &= bad2 == 0

    print("\n=== L2: terminal outcome (uncensored only) ===")
    print(con.execute("""
        SELECT outcome, count(*) AS n, round(100.0*count(*)/sum(count(*)) OVER (), 2) AS pct
        FROM labels_l2 GROUP BY 1 ORDER BY n DESC
    """).df().to_string(index=False))
    pos = con.execute("SELECT round(100.0*avg(CASE WHEN deposited THEN 1 ELSE 0 END), 2) FROM labels_l2").fetchone()[0]
    print(f"positive rate: {pos}%  (spec expects roughly 4%)")

    print("\n=== L3: within-cluster preference pairs ===")
    n3 = con.execute("SELECT count(*) FROM labels_l3").fetchone()[0]
    if n3:
        print(con.execute("""
            SELECT hard, cross_centre, count(*) AS n, round(avg(stage_gap), 2) AS mean_gap
            FROM labels_l3 GROUP BY 1, 2 ORDER BY n DESC
        """).df().to_string(index=False))
        hard = con.execute("SELECT count(*) FROM labels_l3 WHERE hard").fetchone()[0]
        print(f"\ntotal pairs: {n3:,}   hard (>70% identity): {hard:,}   clusters represented: "
              f"{con.execute('SELECT count(DISTINCT cluster_id) FROM labels_l3').fetchone()[0]:,}")
        print(f"hard pairs in the tens of thousands (spec gate): {'PASS' if hard >= 10_000 else 'FAIL'}")
        ok &= hard >= 10_000
        bad3 = con.execute("SELECT count(*) FROM labels_l3 WHERE chosen_stage <= rejected_stage + 1").fetchone()[0]
        print(f"pairs violating the stage gap (must be 0): {bad3}")
        ok &= bad3 == 0
    else:
        print("no pairs built")
        ok = False
    return bool(ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--cap", type=int, default=None, help="max pairs per cluster (default from config)")
    args = ap.parse_args()
    cfg = yaml.safe_load(CFG.read_text())
    cap = args.cap if args.cap is not None else cfg.get("l3", {}).get("max_pairs_per_cluster", 20)

    con = duckdb.connect(str(DB))
    attach_views(con)
    if not args.report:
        build_l1(con)
        build_l2(con)
        build_l3(con, cap)
        print(f"wrote L1, L2, L3 to {PQ}\n")
    ok = report(con)
    print("\nL1/L2/L3:", "PASS" if ok else "FAIL")
    con.close()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
