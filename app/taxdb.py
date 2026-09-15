"""taxdb.py: taxonomy for the web app, read from DuckDB instead of held in memory.

scripts/taxonomy.py loads 3.0 million nodes and 3.5 million names into dictionaries, which
took 4.5 s and several hundred megabytes. In the app that surfaced as a 5,214 ms first
request against 764 ms for the second, paid again by every gunicorn worker, each holding
its own copy on a droplet with under 4 GB. scripts/build_taxonomy_table.py precomputes the
lineages once into `taxonomy_lookup`; this module only queries it.

The resolution ladder is the batch class's, so the app and the training features agree:
taxon id, then the exact name, then progressively shorter prefixes (how "Arabidopsis
thaliana Columbia" resolves, since that exact string is not an NCBI name), then the genus.

If the table has not been built, it falls back to the in-memory class and says so once,
rather than failing a request.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "faffabout.duckdb"

_con: duckdb.DuckDBPyConnection | None = None
_has_table: bool | None = None
_fallback = None

EMPTY = {"taxid_resolved": None, "superkingdom": "", "kingdom": "", "domain": "",
         "phylum": "", "tax_class": "", "tax_order": "", "family": "", "genus": "",
         "resolved_by": ""}
COLS = "taxid, superkingdom, kingdom, phylum, tax_class, tax_order, family, genus"


def _connection() -> duckdb.DuckDBPyConnection:
    # read_only to match every other connection this process opens on the same file:
    # DuckDB refuses a second connection to one path with a different configuration.
    global _con
    if _con is None:
        _con = duckdb.connect(str(DB), read_only=True)
    return _con


def table_present() -> bool:
    global _has_table
    if _has_table is None:
        try