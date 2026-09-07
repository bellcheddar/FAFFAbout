#!/usr/bin/env python
"""03_normalise_status.py: canonical status_canon / stage_ord for every status event.

Reads
  Documentation/targetTrackEnumeratedDataItems-v1.4.1.xls   the authoritative vocabulary
  config/status_map.yaml                                     the canonical mapping
  data/parquet/status_history/*.parquet                      status_raw values in the data

Checks
  * every vocabulary value in the spreadsheet has a mapping entry (hard fail otherwise)
  * every status_raw value seen in the data is mapped, or is listed with its count in
    the unmapped report (never a silent NULL)

Writes
  data/parquet/status_map.parquet     status_raw -> status_canon, stage_ord, method, note, terminal
  data/parquet/status_unmapped.csv    status_raw, n_rows (empty file when everything maps)
  data/parquet/status_history_canon/  status_history joined to the map (one part per input part)
  data/parquet/target_stage/          per target: max_stage, method, n_events, last activity,
                                       stopped flag (the Phase 2 inputs)

Usage
  .venv/bin/python scripts/03_normalise_status.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
XLS = ROOT / "data" / "raw" / "TargetTrack" / "TargetTrack-1Jul2017" / "Documentation" / "targetTrackEnumeratedDataItems-v1.4.1.xls"
MAP_YAML = ROOT / "config" / "status_map.yaml"
PQ = ROOT / "data" / "parquet"


def spreadsheet_vocab() -> list[str]:
    """The 'Status (Target or Trial)' column of the controlled-vocabulary sheet."""
    df = pd.read_excel(XLS, header=None)
    # locate the header cell, then read down its "Current Values" column until the next block
    hits = [(r, c) for r in range(df.shape[0]) for c in range(df.shape[1])
            if str(df.iat[r, c]).strip() == "Status (Target or Trial)"]
    if not hits:
        sys.exit("could not find 'Status (Target or Trial)' in the spreadsheet")
    r0, c0 = hits[0]
    col = c0 + 2  # Data Item | Description | Current Values
    vals = []
    for r in range(r0, df.shape[0]):
        v = df.iat[r, col]
        if isinstance(v, str) and v.strip():
            if v.strip() == "Current Values":
                break
            vals.append(v.strip())
        elif r > r0 and str(df.iat[r, c0]).strip() not in ("nan", "", "Status (Target or Trial)"):
            break  # next Data Item block started
    return vals


def main() -> None:
    cfg = yaml.safe_load(MAP_YAML.read_text())
    smap = cfg["statuses"]
    vocab = spreadsheet_vocab()
    print(f"spreadsheet vocabulary: {len(vocab)} values")
    missing = [v for v in vocab if v not in smap]
    if missing:
        sys.exit(f"!! spreadsheet values with no mapping in {MAP_YAML.name}: {missing}")

    rows = []
    for raw, m in smap.items():
        rows.append({
            "status_raw": raw, "status_canon": m["canon"], "stage_ord": m.get("stage_ord"),
            "method": m.get("method", "common"), "note": m.get("note", ""),
            "terminal": bool(m.get("terminal", False)),
            "in_spreadsheet": raw in vocab,
        })
    map_df = pd.DataFrame(rows)
    map_df["stage_ord"] = map_df["stage_ord"].astype("Int8")
    map_df.to_parquet(PQ / "status_map.parquet", index=False)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW sh AS SELECT * FROM read_parquet('{PQ}/status_history/*.parquet')")
    con.execute(f"CREATE VIEW smap AS SELECT * FROM read_parquet('{PQ}/status_map.parquet')")

    seen = con.execute("SELECT status_raw, count(*) n FROM sh GROUP BY 1 ORDER BY 2 DESC").df()
    unmapped = seen[~seen.status_raw.isin(map_df.status_raw)]
    unmapped.to_csv(PQ / "status_unmapped.csv", index=False)
    print(f"distinct status_raw in data: {len(seen)}  mapped: {len(seen) - len(unmapped)}  unmapped: {len(unmapped)}")
    if len(unmapped):
        print("!! unmapped values (kept as status_canon = NULL, listed in status_unmapped.csv):")
        print(unmapped.to_string(index=False))

    # joined status history, partitioned like the input
    out = PQ / "status_history_canon"
    out.mkdir(exist_ok=True)
    con.execute(f"""
        COPY (
          SELECT sh.*, smap.status_canon, smap.stage_ord, smap.method, smap.note, smap.terminal
          FROM sh LEFT JOIN smap USING (status_raw)
        ) TO '{out / 'status_history_canon.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
    """)

    # per-target stage summary
    ts = PQ / "target_stage"
    ts.mkdir(exist_ok=True)
    con.execute(f"""
        COPY (
          WITH c AS (SELECT * FROM read_parquet('{out}/*.parquet')),
          agg AS (
            SELECT target_id, centre,
                   max(stage_ord)                                    AS max_stage,
                   count(*)                                          AS n_events,
                   count(*) FILTER (WHERE stage_ord IS NOT NULL)     AS n_ladder_events,
                   min(NULLIF(status_date, ''))                      AS first_event,
                   max(NULLIF(status_date, ''))                      AS last_event,
                   bool_or(terminal)                                 AS stopped,
                   max(CASE WHEN method = 'xray' THEN 1 ELSE 0 END)  AS any_xray,
                   max(CASE WHEN method = 'nmr'  THEN 1 ELSE 0 END)  AS any_nmr,
                   max(CASE WHEN method = 'em'   THEN 1 ELSE 0 END)  AS any_em,
                   count(DISTINCT trial_id)                          AS n_trials,
                   count(*) FILTER (WHERE status_canon IS NULL)      AS n_unmapped
            FROM c GROUP BY 1, 2)
          SELECT *, CASE WHEN any_xray = 1 THEN 'xray' WHEN any_nmr = 1 THEN 'nmr'
                         WHEN any_em = 1 THEN 'em' ELSE 'none' END AS method
          FROM agg
        ) TO '{ts / 'target_stage.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    print("\nstage_ord distribution over status events:")
    print(con.execute(f"""
        SELECT coalesce(CAST(stage_ord AS VARCHAR), 'non-ladder') AS stage_ord, status_canon, count(*) n
        FROM read_parquet('{out}/*.parquet') GROUP BY 1, 2 ORDER BY stage_ord NULLS LAST, n DESC
    """).df().to_string(index=False))
    print("\nmax_stage per target:")
    print(con.execute(f"""
        SELECT coalesce(CAST(max_stage AS VARCHAR), 'none') AS max_stage, count(*) n_targets,
               round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
        FROM read_parquet('{ts}/*.parquet') GROUP BY 1 ORDER BY max_stage NULLS LAST
    """).df().to_string(index=False))
    print("\nNULL status_canon rows:",
          con.execute(f"SELECT count(*) FROM read_parquet('{out}/*.parquet') WHERE status_canon IS NULL").fetchone()[0])


if __name__ == "__main__":
    main()
