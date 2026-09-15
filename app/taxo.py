"""taxo.py: taxonomy lookup for the serving path, out of DuckDB rather than out of RAM.

scripts/taxonomy.py loads nodes.dmp and names.dmp into dictionaries: 3.0M nodes and 3.5M
names, 4.5 s and several hundred megabytes. Right for a batch pass over the archive, wrong
for a web worker, where it showed up as a 5,214 ms first request against 764 ms for the
second, repeated per gunicorn worker, on a droplet with under 4 GB.

Same resolution ladder as the batch class, so the app and the training pipeline agree:
exact name, then progressively shorter prefixes ("Arabidopsis thaliana Columbia" has no
exact key and must fall back), then the bare genus.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "faffabout.duckdb"

EMPTY = {"taxid_resolved": None, "superkingdom": "", "kingdom": "", "phylum": "",
         "tax_class": "", "tax_order": "", "family": "", "genus": "", "resolved_by": ""}
COLS = ("taxid", "superkingdom", "kingdom", "phylum", "tax_class", "tax_order", "family", "genus")

_CON: duckdb.DuckDBPyConnection | None = None


def con() -> duckdb.DuckDBPyConnection:
    global _CON
    if _CON is None:
        _CON = duckdb.connect(str(DB), read_only=True)
    return _CON


def available() -> bool:
    try:
        con().execute("SELECT 1 FROM taxonomy_lookup LIMIT 1").fetchone()
        return True
    except Exception:  # noqa: BLE001
        return False


def _row_to_dict(row, how: str) -> dict:
    d = dict(zip(COLS, row))
    return {"taxid_resolved": d["taxid"], "superkingdom": d["superkingdom"] or "",
            "kingdom": d["kingdom"] or "", "phylum": d["phylum"] or "",
            "tax_class": d["tax_class"] or "", "tax_order": d["tax_order"] or "",
            "family": d["family"] or "", "genus": d["genus"] or "", "resolved_by": how}


@lru_cache(maxsize=20_000)
def by_id(taxid: str | int) -> dict | None:
    try:
        t = int(str(taxid).strip())
    except (TypeError, ValueError):
        return None
    r = con().execute(
        f"SELECT {', '.join(COLS)} FROM taxonomy_by_id WHERE taxid = ?", [t]).fetchone()
    return _row_to_dict(r, "id") if r else None


@lru_cache(maxsize=20_000)
def by_name(organism: str) -> dict | None:
    if not organism or not organism.strip():
        return None
    words = organism.strip().split()
    # exact, then shorter prefixes (dropping strain suffixes), then the genus alone
    keys = [" ".join(words[:k]).casefold() for k in range(len(words), 1, -1)]
    if words:
        keys.append(words[0].casefold())
    rows = con().execute(
        f"SELECT name_key, {', '.join(COLS)} FROM taxonomy_lookup WHERE name_key IN "
        f"({','.join('?' * len(keys))})", keys).fetchall()
    found = {r[0]: r[1:] for r in rows}
    for k in keys:                       # keys are already longest-first
        if k in found:
            return _row_to_dict(found[k], "name")
    return None


def classify(taxid=None, organism: str = "") -> dict:
    """Best effort from an id, a name, or both. Never raises."""
    try:
        return by_id(taxid) or by_name(organism) or dict(EMPTY)
    except Exception:  # noqa: BLE001
        return dict(EMPTY)
