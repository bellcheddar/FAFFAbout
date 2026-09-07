"""taxonomy.py: NCBI taxonomy lookup for kingdom and lineage features.

The archive records an organism NAME for 99.4% of targets but a taxon ID for only 44.4%,
so resolving by name matters more than resolving by id. Names are matched case-folded
after stripping the strain suffixes the archive carries ("Arabidopsis thaliana Columbia",
"Pyrococcus furiosus DSM 3638"), falling back to the genus.

Source: ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz, extracted to data/external/.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"

# NCBI renamed the top-level rank from "superkingdom" to "domain" and inserted new
# kingdom-level clades beneath it (Metazoa, Viridiplantae, Bacillati, Pseudomonadati),
# so reading the rank label alone returns "Metazoa" where "Eukaryota" is meant. These
# four names are matched anywhere in the lineage instead, which survives the next
# reshuffle as well. Viruses sit at rank "acellular root", not a domain at all.
SUPERKINGDOMS = ("Bacteria", "Archaea", "Eukaryota", "Viruses")
WANTED_RANKS = ("domain", "superkingdom", "kingdom", "phylum", "class", "order", "family", "genus")


class Taxonomy:
    def __init__(self, ext: Path = EXT):
        self.parent: dict[int, int] = {}
        self.rank: dict[int, str] = {}
        self.name: dict[int, str] = {}
        self.by_name: dict[str, int] = {}
        self.merged: dict[int, int] = {}
        self._load(ext)

    def _load(self, ext: Path) -> None:
        with (ext / "nodes.dmp").open(encoding="latin-1") as fh:
            for line in fh:
                f = line.split("\t|\t")
                tid, par, rank = int(f[0]), int(f[1]), f[2]
                self.parent[tid] = par
                self.rank[tid] = rank
        with (ext / "names.dmp").open(encoding="latin-1") as fh:
            for line in fh:
                f = line.split("\t|")
                tid, nm, cls = int(f[0]), f[1].strip(), f[3].strip().rstrip("\t|").strip()
                if cls == "scientific name":
                    self.name[tid] = nm
                    self.by_name.setdefault(nm.casefold(), tid)
                elif cls in ("synonym", "equivalent name", "genbank synonym"):
                    self.by_name.setdefault(nm.casefold(), tid)
        mp = ext / "merged.dmp"
        if mp.exists():
            with mp.open(encoding="latin-1") as fh:
                for line in fh:
                    f = line.split("\t|")
                    self.merged[int(f[0])] = int(f[1])

    def resolve_id(self, taxid) -> int | None:
        try:
            t = int(str(taxid).strip())
        except (TypeError, ValueError):
            return None
        t = self.merged.get(t, t)
        return t if t in self.parent else None

    def resolve_name(self, organism: str) -> int | None:
        """Full name, then progressively shorter prefixes, then the bare genus."""
        if not organism:
            return None
        s = organism.strip()
        key = s.casefold()
        if key in self.by_name:
            return self.by_name[key]
        words = s.split()
        for k in range(len(words) - 1, 1, -1):       # drop strain suffixes
            cand = " ".join(words[:k]).casefold()
            if cand in self.by_name:
                return self.by_name[cand]
        if words:                                     # genus only
            return self.by_name.get(words[0].casefold())
        return None

    @lru_cache(maxsize=200_000)
    def lineage(self, taxid: int) -> tuple[tuple[str, str], ...]:
        out, seen, t = [], set(), taxid
        while t and t not in seen and t != 1:
            seen.add(t)
            out.append((self.rank.get(t, "no rank"), self.name.get(t, "")))
            t = self.parent.get(t, 0)
        return tuple(out)

    def ranks_of(self, taxid: int | None) -> dict[str, str]:
        if taxid is None:
            return {}
        d = {}
        for r, n in self.lineage(taxid):
            if r in WANTED_RANKS:
                d.setdefault(r, n)
        return d

    def classify(self, taxid=None, organism: str = "") -> dict:
        """Best-effort kingdom and lineage from an id, a name, or both."""
        t = self.resolve_id(taxid) or self.resolve_name(organism)
        r = self.ranks_of(t)
        # match by name anywhere in the lineage, not by rank label
        names = {n for _, n in self.lineage(t)} if t else set()
        sk = next((k for k in SUPERKINGDOMS if k in names), "")
        return {
            "taxid_resolved": t,
            "superkingdom": sk,
            "kingdom": r.get("kingdom", ""),
            "domain": r.get("domain", "") or r.get("superkingdom", ""),
            "phylum": r.get("phylum", ""),
            "tax_class": r.get("class", ""),
            "tax_order": r.get("order", ""),
            "family": r.get("family", ""),
            "genus": r.get("genus", ""),
            "resolved_by": "id" if self.resolve_id(taxid) else ("name" if t else ""),
        }
