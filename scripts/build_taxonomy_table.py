#!/usr/bin/env python
"""build_taxonomy_table.py: flatten the NCBI dumps into one DuckDB lookup table.

Why this exists. `scripts/taxonomy.py` loads nodes.dmp and names.dmp into dictionaries:
3.0 million nodes and 3.5 million names, which takes 4.5 seconds and several hundred
megabytes. That is right for a one-off batch pass over the archive, and wrong for a web
worker. Measured in the app, the first request paid 5,214 ms against 764 ms for the
second, and every gunicorn worker would pay it again and hold its own copy. The droplet
has under 4 GB.

So the lookup is precomputed once into `taxonomy_lookup`, keyed on a casefolded name, and
queried read-only at request time: no per-worker memory and no start-up cost.

The table keeps the same resolution ladder the batch class uses, so the app and the
training pipeline agree: exact name, then progressively shorter prefixes (which is how
"Arabidopsis thaliana Columbia" resolves), then the bare genus.

Usage
  .venv/bin/python scripts/build_taxonomy_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from taxonomy import SUPERKINGDOMS, Taxonomy  # noqa: E402

DB = ROOT / "data" / "faffabout.duckdb"
OUT = ROOT / "data" / "parquet" / "taxonomy_lookup.parquet"


def main() -> None:
    print("loading the NCBI dumps (this is the cost being removed from the app) ...")
    tx = Taxonomy()
    print(f"  {len(tx.parent):,} nodes, {len(tx.by_name):,} names")

    # Resolve every node once, then key by every name that points at it.
    print("resolving lineages ...")
    per_taxid: dict[int, dict] = {}
    for tid in tx.name:
        lin = tx.lineage(tid)
        names = {n for _, n in lin}
        ranks: dict[str, str] = {}
        for r, n in lin:
            ranks.setdefault(r, n)
        per_taxid[tid] = {
            "superkingdom": next((k for k in SUPERKINGDOMS if k in names), ""),
            "kingdom": ranks.get("kingdom", ""),
            "phylum": ranks.get("phylum", ""),
            "tax_class": ranks.get("class", ""),
            "tax_order": ranks.get("order", ""),
            "family": ranks.get("family", ""),
            "genus": ranks.get("genus", ""),
        }

    print("building the name index ...")
    rows = []
    for key, tid in tx.by_name.items():
        d = per_taxid.get(tid)
        if d is None:
            continue
        rows.append({"name_key": key, "taxid": tid, **d})
    df = pd.DataFrame(rows).drop_duplicates("name_key")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"  {len(df):,} name keys -> {OUT} ({OUT.stat().st_size / 1e6:.0f} MB)")

    con = duckdb.connect(str(DB))
    con.execute(f"""
        CREATE OR REPLACE TABLE taxonomy_lookup AS
        SELECT * FROM read_parquet('{OUT}')
    """)
    con.execute("CREATE OR REPLACE TABLE taxonomy_by_id AS "
                "SELECT DISTINCT taxid, superkingdom, kingdom, phylum, tax_class, "
                "tax_order, family, genus FROM taxonomy_lookup")
    n, n_id = con.execute(
        "SELECT (SELECT count(*) FROM taxonomy_lookup), (SELECT count(*) FROM taxonomy_by_id)"
    ).fetchone()
    print(f"wrote taxonomy_lookup ({n:,} rows) and taxonomy_by_id ({n_id:,} rows) into {DB.name}")

    # A check that would have caught the rank-label bug: these must be domains, not kingdoms.
    print("\nspot check:")
    for name in ("escherichia coli", "homo sapiens", "pyrococcus furiosus",
                 "arabidopsis thaliana columbia", "leishmania major"):
        r = con.execute("SELECT superkingdom, genus FROM taxonomy_lookup WHERE name_key = ?",
                        [name]).fetchone()
        print(f"  {name:<32} {r[0] if r else '(absent, needs prefix fallback)':<11}"
              f"{r[1] if r else ''}")
    con.close()


if __name__ == "__main__":
    main()
