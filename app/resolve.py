"""resolve.py: turn whatever the user pasted into a sequence, and say what it was.

One field accepts three things (spec section 7.2):

  raw FASTA            with or without a header, whitespace and digits tolerated
  UniProt accession    P0A6Y8, A0A023GPI8, or P0A6Y8-2 (isoform suffix stripped)
  PDB ID               4XB7, or 4XB7_A / 4xb7.A / 4XB7:A with a chain

The resolved sequence is ALWAYS returned for display before anything is computed, because
a silently mis-resolved identifier produces a confident forecast for the wrong protein.

The two lookups are deliberately different services: EBI/UniProt for accessions, RCSB for
PDB entries. Both were verified against P0A6Y8 (DnaK, 638 aa) and 4XB7 chain A (400 aa).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import requests

UNIPROT_RE = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?$")
PDB_RE = re.compile(r"^([1-9][A-Za-z0-9]{3})(?:[_.:\-]([A-Za-z0-9]{1,4}))?$")
AA_OK = set("ACDEFGHIKLMNPQRSTVWYBXZUO*")

# The twenty that carry a value on every composition scale. B (Asx), Z (Glx), X (unknown),
# U (selenocysteine), O (pyrrolysine) and * (stop) are accepted into the sequence but
# contribute to no scale, so a sequence made only of those divides by zero.
#
# Measured 2026-09-16: an all-X 30-mer is accepted by from_fasta and yields EIGHT NaN
# features (pI, GRAVY, net charge and all five composition fractions). NaN does not raise,
# it renders, so the user gets a 200 with "pI: nan" in the prompt, and the booster receives
# 11 of its 37 features as missing and answers from a learned default split: a confident
# forecast computed from no sequence information at all. The boundary is exact - ONE
# standard residue among thirty-nine X's makes every scale finite - so this is a single
# empty-denominator case, not a general problem with masked sequences.
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def n_standard(seq: str) -> int:
    """How many residues actually contribute to the composition scales."""
    return sum(c in STANDARD_AA for c in seq)


# Twenty SCOREABLE residues, matching the twenty-residue floor from_fasta already applies.
# Every composition feature is computed over standard residues alone, so a 40-mer carrying
# one methionine rests on exactly the same statistical basis as a 1-mer: rejecting only the
# zero case fixes the NaN and still lets a forecast be built on almost nothing.
MIN_SCOREABLE = 20

# Above this share of placeholders the features are honest but thin, so say so rather than
# refuse: UniProt entries with a few uncertain residues and PDB constructs carrying some UNK
# are ordinary inputs, not mistakes.
PLACEHOLDER_WARN_FRAC = 0.2


def assert_scoreable(seq: str) -> list[str]:
    """Refuse a sequence no feature can honestly be computed from; warn about a thin one.

    Called from all three entry points rather than from_fasta alone: from_uniprot and
    from_pdb build Resolved directly, and a PDB chain of UNK residues maps to poly-X, which
    is a routine low-resolution model rather than a pathological paste.

    Returns warnings so the caller can surface them; raises only on the hard floor.
    """
    n, total = n_standard(seq), len(seq)
    if n < MIN_SCOREABLE:
        raise ResolveError(
            f"only {n} of {total} residues are standard amino acids, and at least "
            f"{MIN_SCOREABLE} are needed. pI, hydropathy, charge and composition are "
            "computed from those residues alone, so a forecast would rest on almost nothing.")
    if total and (total - n) / total > PLACEHOLDER_WARN_FRAC:
        return [f"{total - n} of {total} residues are placeholders (X, B, Z, U, O or *); "
                f"every composition feature is computed from the other {n}"]
    return []

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{acc}.json"
RCSB_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{pdb}"
RCSB_ENTITY = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb}/{entity}"
TIMEOUT = 30


class ResolveError(ValueError):
    """The input could not be turned into a protein sequence."""


@dataclass
class Resolved:
    sequence: str
    source: str                      # fasta | uniprot | pdb
    accession: str = ""
    name: str = ""
    organism: str = ""
    taxon_id: str = ""
    chain: str = ""
    header: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.sequence)

    def as_dict(self) -> dict:
        return {"source": self.source, "accession": self.accession, "name": self.name,
                "organism": self.organism, "taxon_id": self.taxon_id, "chain": self.chain,
                "length": self.length, "sequence": self.sequence, "warnings": self.warnings}


def clean_sequence(raw: str) -> tuple[str, list[str]]:
    """Strip whitespace, digits and gaps; uppercase; report anything unexpected."""
    warnings: list[str] = []
    body = re.sub(r"[\s\d\-\*\.]", "", raw).upper()
    unknown = sorted({c for c in body if c not in AA_OK})
    if unknown:
        warnings.append(f"ignored {len(unknown)} unrecognised character(s): {''.join(unknown)}")
        body = "".join(c for c in body if c in AA_OK)
    # U (selenocysteine) and O (pyrrolysine) are real but break most feature scales
    for odd, what in (("U", "selenocysteine"), ("O", "pyrrolysine"), ("B", "Asx"), ("Z", "Glx")):
        if odd in body:
            warnings.append(f"contains {what} ({odd}); treated as unknown in the feature calculations")
    return body, warnings


def sniff(text: str) -> str:
    """fasta | uniprot | pdb, from the shape of the input alone."""
    s = text.strip()
    if not s:
        raise ResolveError("nothing entered")
    if s.startswith(">"):
        return "fasta"
    token = s.split()[0] if len(s.split()) == 1 else ""
    if token:
        if UNIPROT_RE.match(token.upper()):
            return "uniprot"
        if PDB_RE.match(token):
            return "pdb"
    # a bare sequence: mostly amino-acid letters and long enough to mean something
    letters = re.sub(r"[\s\d\-\*\.]", "", s).upper()
    if len(letters) >= 20 and sum(c in AA_OK for c in letters) / len(letters) > 0.9:
        return "fasta"
    raise ResolveError(
        "could not tell whether that is a sequence, a UniProt accession or a PDB ID. "
        "Paste a FASTA sequence, an accession such as P0A6Y8, or a PDB ID such as 4XB7_A.")


def from_fasta(text: str) -> Resolved:
    lines = text.strip().splitlines()
    header = lines[0][1:].strip() if lines and lines[0].startswith(">") else ""
    body = "\n".join(lines[1:] if header else lines)
    seq, warnings = clean_sequence(body)
    if len(seq) < 20:
        raise ResolveError(f"sequence is only {len(seq)} residues; at least 20 are needed")
    warnings += assert_scoreable(seq)
    name = header.split("|")[-1].strip() if header else ""
    return Resolved(sequence=seq, source="fasta", name=name, header=header, warnings=warnings)


def from_uniprot(token: str, session: requests.Session | None = None) -> Resolved:
    acc = token.strip().upper()
    base = acc.split("-")[0]           # an isoform suffix is not part of the accession path
    get = (session or requests).get
    r = get(UNIPROT_URL.format(acc=base), timeout=TIMEOUT)
    if r.status_code == 404:
        raise ResolveError(f"UniProt has no entry for {acc}")
    r.raise_for_status()
    d = r.json()
    seq, warnings = clean_sequence(d.get("sequence", {}).get("value", ""))
    if not seq:
        raise ResolveError(f"{acc} carries no sequence")
    warnings += assert_scoreable(seq)
    if acc != base:
        warnings.append(f"isoform suffix ignored: resolved the canonical sequence for {base}")
    desc = d.get("proteinDescription", {})
    name = (desc.get("recommendedName", {}).get("fullName", {}).get("value")
            or (desc.get("submissionNames") or [{}])[0].get("fullName", {}).get("value") or "")
    org = d.get("organism", {})
    return Resolved(sequence=seq, source="uniprot", accession=base, name=name,
                    organism=org.get("scientificName", ""), taxon_id=str(org.get("taxonId", "") or ""),
                    warnings=warnings)


def from_pdb(token: str, session: requests.Session | None = None) -> Resolved:
    m = PDB_RE.match(token.strip())
    if not m:
        raise ResolveError(f"{token} is not a PDB ID")
    pdb, chain = m.group(1).upper(), (m.group(2) or "").upper()
    get = (session or requests).get
    r = get(RCSB_ENTRY.format(pdb=pdb), timeout=TIMEOUT)
    if r.status_code == 404:
        raise ResolveError(f"the PDB has no entry {pdb}")
    r.raise_for_status()
    entities = (r.json().get("rcsb_entry_container_identifiers", {})
                .get("polymer_entity_ids") or [])
    if not entities:
        raise ResolveError(f"{pdb} contains no polymer entity")

    warnings: list[str] = []
    chosen = None
    for ent in entities:
        er = get(RCSB_ENTITY.format(pdb=pdb, entity=ent), timeout=TIMEOUT)
        er.raise_for_status()
        ed = er.json()
        ids = ed.get("rcsb_polymer_entity_container_identifiers", {})
        auth = [c.upper() for c in (ids.get("auth_asym_ids") or [])]
        poly = ed.get("entity_poly", {}) or {}
        # one-letter *canonical* code: the modified-residue form contains (XYZ) groups
        seq_raw = poly.get("pdbx_seq_one_letter_code_can") or ""
        if poly.get("rcsb_entity_polymer_type") not in (None, "Protein"):
            continue
        if chain and chain not in auth:
            continue
        chosen = (ent, auth, seq_raw, ed)
        break
    if chosen is None:
        raise ResolveError(
            f"{pdb} has no protein chain {chain}" if chain else f"{pdb} has no protein entity")

    ent, auth, seq_raw, ed = chosen
    seq, w = clean_sequence(seq_raw)
    warnings += w
    # The most reachable route of the three: a chain of UNK residues maps to poly-X, which
    # is a routine low-resolution model rather than a pathological paste.
    warnings += assert_scoreable(seq)
    if not chain and len(entities) > 1:
        warnings.append(f"{pdb} has {len(entities)} entities; resolved entity {ent} "
                        f"(chain {'/'.join(auth)}). Append a chain, e.g. {pdb}_{auth[0]}, to pick another")
    warnings.append("a PDB sequence is the construct that crystallised, expression tags and all, "
                    "which is not necessarily the construct you would order")
    src = ed.get("rcsb_polymer_entity", {})
    org = (ed.get("rcsb_entity_source_organism") or [{}])[0]
    return Resolved(sequence=seq, source="pdb", accession=pdb, chain=chain or (auth[0] if auth else ""),
                    name=src.get("pdbx_description", "") or "",
                    organism=org.get("scientific_name", "") or "",
                    taxon_id=str(org.get("ncbi_taxonomy_id", "") or ""), warnings=warnings)


def resolve(text: str, session: requests.Session | None = None) -> Resolved:
    kind = sniff(text)
    if kind == "fasta":
        return from_fasta(text)
    if kind == "uniprot":
        return from_uniprot(text, session)
    return from_pdb(text, session)
