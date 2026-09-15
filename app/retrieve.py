"""retrieve.py: find the archive precedents for a query sequence, and score their fate.

MMseqs2 search against every distinct sequence in the archive, then a DuckDB join onto
targets, censoring and outcomes. Measured: 0.83 s wall for a 638-residue query against
300,027 sequences on eight threads, so this is cheap enough to do on every request.

The asymmetry worth knowing about. At TRAINING time, archive context came from a target's
MMseqs2 CLUSTER membership. At inference the query is not in any cluster, so context comes
from a SEARCH instead. The two are made to agree by using the clustering's own thresholds:
30% identity and 80% coverage of the shorter sequence (`--min-seq-id 0.30 -c 0.8
--cov-mode 1` in scripts/04_cluster_sequences.sh). Precedents above 70% identity are the
`close` stratum, matching tt70.

Censored precedents are RETAINED and flagged, never dropped: that is spec rule 1, and the
interface renders them grey and dashed rather than red.
"""
from __future__ import annotations

import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "faffabout.duckdb"
SEARCH_DB = ROOT / "data" / "search" / "archiveDB"

# Mirrors scripts/04_cluster_sequences.sh so search-time context matches training-time context.
MIN_IDENTITY = 0.30
MIN_COVERAGE = 0.80
CLOSE_IDENTITY = 0.70
MAX_HITS = 2000
EVALUE = 1e-3
N_GATES = 8

FORMAT = "query,target,fident,alnlen,qcov,tcov,evalue,bits"


@dataclass
class Precedent:
    target_id: str
    centre: str
    organism: str
    identity: float
    coverage: float
    max_stage: int | None
    censored: bool
    censored_reason: str
    method: str
    n_pdb: int
    first_deposit: str
    stop_status: str
    host: str
    tag: str
    note: str

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["identity"] = round(self.identity, 3)
        d["coverage"] = round(self.coverage, 3)
        return d


def md5(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()


def search(sequence: str, threads: int = 8, max_hits: int = MAX_HITS) -> list[dict]:
    """MMseqs2 easy-search of one sequence against the archive."""
    if not SEARCH_DB.with_suffix(".dbtype").exists() and not Path(str(SEARCH_DB) + ".dbtype").exists():
        raise FileNotFoundError(
            f"no search database at {SEARCH_DB}. Build it with:\n"
            f"  mmseqs createdb data/parquet/targets.fasta {SEARCH_DB}\n"
            f"  mmseqs createindex {SEARCH_DB} tmp")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fa, out = tmp / "q.fa", tmp / "hits.m8"
        fa.write_text(f">query\n{sequence}\n")
        cmd = ["mmseqs", "easy-search", str(fa), str(SEARCH_DB), str(out), str(tmp / "t"),
               "--threads", str(threads), "-v", "1", "--max-seqs", str(max_hits),
               "-e", str(EVALUE), "--format-output", FORMAT]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"mmseqs failed: {proc.stderr[-400:]}")
        rows = []
        for line in out.read_text().splitlines():
            f = line.split("\t")
            if len(f) < 8:
                continue
            rows.append({"seq_md5": f[1], "identity": float(f[2]), "alnlen": int(f[3]),
                         "qcov": float(f[4]), "tcov": float(f[5]),
                         "evalue": float(f[6]), "bits": float(f[7])})
    return rows


def precedents(sequence: str, con: duckdb.DuckDBPyConnection | None = None,
               threads: int = 8, exclude_self: bool = True) -> list[Precedent]:
    """Search, filter to the clustering thresholds, and join onto the archive tables."""
    hits = search(sequence, threads=threads)
    q_md5 = md5(sequence)
    keep = {}
    for h in hits:
        if h["identity"] < MIN_IDENTITY:
            continue
        # coverage of the SHORTER sequence, matching --cov-mode 1
        if max(h["qcov"], h["tcov"]) < MIN_COVERAGE:
            continue
        if exclude_self and h["seq_md5"] == q_md5:
            continue
        prev = keep.get(h["seq_md5"])
        if prev is None or h["identity"] > prev["identity"]:
            keep[h["seq_md5"]] = h
    if not keep:
        return []

    own = con or duckdb.connect(str(DB), read_only=True)
    try:
        own.execute("CREATE OR REPLACE TEMP TABLE _hits (seq_md5 VARCHAR, identity DOUBLE, coverage DOUBLE)")
        own.executemany("INSERT INTO _hits VALUES (?, ?, ?)",
                        [(k, v["identity"], max(v["qcov"], v["tcov"])) for k, v in keep.items()])
        rows = own.execute("""
            SELECT t.target_id, t.centre, t.organism, h.identity, h.coverage,
                   c.max_stage, coalesce(c.censored, false) AS censored,
                   coalesce(c.censored_reason, '') AS censored_reason,
                   coalesce(c.method, '') AS method,
                   coalesce(o.n_pdb, 0) AS n_pdb, coalesce(o.first_deposit, '') AS first_deposit,
                   coalesce(t.stop_status, '') AS stop_status,
                   coalesce(f.host, '') AS host, coalesce(f.tag, '') AS tag
            FROM _hits h
            JOIN targets t USING (seq_md5)
            LEFT JOIN censoring c USING (target_id)
            LEFT JOIN features f USING (target_id)
            LEFT JOIN (SELECT target_id, count(DISTINCT pdb_id) n_pdb,
                              min(NULLIF(deposit_date, '')) first_deposit
                       FROM outcomes GROUP BY 1) o USING (target_id)
            WHERE c.max_stage IS NOT NULL
            ORDER BY h.identity DESC, c.max_stage DESC
        """).fetchall()
    finally:
        if con is None:
            own.close()

    out = []
    for (tid, centre, org, ident, cov, stage, cens, reason, method,
         n_pdb, dep, stop, host, tag) in rows:
        out.append(Precedent(
            target_id=tid, centre=centre, organism=org or "", identity=ident, coverage=cov,
            max_stage=int(stage) if stage is not None else None, censored=bool(cens),
            censored_reason=reason, method=method, n_pdb=int(n_pdb), first_deposit=dep or "",
            stop_status=stop, host=host, tag=tag, note=_note(cens, reason, stop, n_pdb)))
    return out


def _note(censored, reason, stop_status, n_pdb) -> str:
    if censored:
        return ("censored: the centre's file closed in bulk, not on a result"
                if reason in ("bulk_closure", "both") else
                "censored: still open when the centre wound down")
    if n_pdb:
        return f"deposited ({n_pdb} structure{'s' if n_pdb > 1 else ''})"
    return stop_status or ""


def context(precs: list[Precedent]) -> dict:
    """The archive-context features, computed the same way training computed them.

    Censored precedents are excluded from the base RATES, because they are evidence of
    nothing, but counted in `n_censored` and shown in the interface.
    """
    usable = [p for p in precs if not p.censored and p.max_stage is not None]
    close = [p for p in precs if p.identity >= CLOSE_IDENTITY and not p.censored]
    n, n_close = len(usable), len(close)
    per_gate = {}
    for g in range(N_GATES):
        entered = [p for p in usable if (p.max_stage or 0) >= g]
        cleared = [p for p in entered if (p.max_stage or 0) > g]
        per_gate[g] = {"n_precedents": len(entered),
                       "n_precedents_cleared": len(cleared),
                       "cluster_base_rate": (len(cleared) / len(entered)) if entered else None}
    deposited = sum(1 for p in usable if (p.max_stage or 0) >= 8)
    return {
        "n_precedents": n,
        "n_close_precedents": n_close,
        "n_censored": sum(1 for p in precs if p.censored),
        "closest_identity": max((p.identity for p in precs), default=0.0),
        "closest_identity_band": 0.70 if n_close else (0.30 if n else 0.0),
        "cluster_censored_frac": (sum(1 for p in precs if p.censored) / len(precs)) if precs else None,
        "cluster_deposit_rate": (deposited / n) if n else None,
        "n_cluster_precedents": n,
        "per_gate": per_gate,
        "strength": _strength(n, n_close),
    }


def _strength(n: int, n_close: int) -> str:
    if n >= 20 and n_close >= 3:
        return "STRONG"
    if n >= 5:
        return "MODERATE"
    if n >= 1:
        return "WEAK"
    return "NONE"
