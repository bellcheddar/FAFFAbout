#!/usr/bin/env python
"""fetch_pdb_metadata.py: fill outcomes.resolution (and method where missing) from RCSB.

targetpdb_info.csv carries method and dates but no resolution. This pulls resolution,
experimental method and release date for every distinct pdb_id in data/parquet/outcomes
through the RCSB GraphQL endpoint in batches of 200, caches to
data/parquet/pdb_metadata.parquet, and rewrites outcomes/outcomes.parquet with the
resolution column filled.

Usage
  .venv/bin/python scripts/fetch_pdb_metadata.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PQ = ROOT / "data" / "parquet"
CACHE = PQ / "pdb_metadata.parquet"
OUTCOMES = PQ / "outcomes" / "outcomes.parquet"
GQL = "https://data.rcsb.org/graphql"
BATCH = 200

QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    exptl { method }
    rcsb_entry_info { resolution_combined }
    rcsb_accession_info { initial_release_date deposit_date }
    struct { title }
  }
}
"""


def fetch(ids: list[str]) -> list[dict]:
    for attempt in range(5):
        try:
            r = requests.post(GQL, json={"query": QUERY, "variables": {"ids": ids}}, timeout=60)
            r.raise_for_status()
            data = r.json()["data"]["entries"] or []
            break
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"  retry {attempt + 1} after {wait}s: {e}")
            time.sleep(wait)
    else:
        return []
    rows = []
    for e in data:
        if not e:
            continue
        res = (e.get("rcsb_entry_info") or {}).get("resolution_combined") or []
        rows.append({
            "pdb_id": e["rcsb_id"].upper(),
            "method": "|".join(m.get("method", "") for m in (e.get("exptl") or [])),
            "resolution": float(res[0]) if res else None,
            "deposit_date": ((e.get("rcsb_accession_info") or {}).get("deposit_date") or "")[:10],
            "release_date": ((e.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10],
            "title": (e.get("struct") or {}).get("title") or "",
        })
    return rows


def main() -> None:
    out = pd.read_parquet(OUTCOMES)
    ids = sorted(set(out.pdb_id.dropna()))
    cache = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=["pdb_id"])
    todo = [i for i in ids if i not in set(cache.pdb_id)]
    print(f"distinct pdb ids: {len(ids):,}  cached: {len(cache):,}  to fetch: {len(todo):,}")
    rows = []
    for i in range(0, len(todo), BATCH):
        rows.extend(fetch(todo[i:i + BATCH]))
        print(f"  {min(i + BATCH, len(todo)):,}/{len(todo):,}", end="\r", flush=True)
        time.sleep(0.2)
    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("pdb_id")
        cache.to_parquet(CACHE, index=False)
    print(f"\nmetadata rows: {len(cache):,}  with resolution: {cache.resolution.notna().sum():,}")

    m = cache.set_index("pdb_id")
    out["resolution"] = out.pdb_id.map(m["resolution"]).astype("float32")
    for col in ("method", "deposit_date", "release_date", "title"):
        fill = out.pdb_id.map(m[col]).fillna("")
        out[col] = out[col].where(out[col].astype(str).str.strip() != "", fill)
    out.to_parquet(OUTCOMES, index=False, compression="zstd")
    missing = out[out.resolution.isna() & out.method.str.contains("X-RAY", na=False)]
    print(f"outcomes rewritten: {len(out):,} rows; X-ray rows still lacking resolution: {len(missing):,}")


if __name__ == "__main__":
    main()
