#!/usr/bin/env python
"""02_parse_tt_xml.py: stream the TargetTrack XML into Parquet tables.

Element names below were verified against Documentation/targetTrack-v1.4.1.xsd and a
real <target> record on 2026-09-07 (see the ELEMENT MAP block). tt.xml.gz is 1.45 GB
decompressed, so it is streamed with lxml.iterparse and cleared element by element,
never DOM-parsed.

Two equivalent inputs (verified: both hold exactly 335,771 <target> elements):
  * TargetsbyContributor/*.xml.gz  43 per-centre files, parsed in parallel (default)
  * TargetTrack XML files/tt.xml.gz one stream, single process (--single)

Outputs (Parquet, 50,000-row groups, one part file per input file):
  data/parquet/targets/           one row per target
  data/parquet/target_sequences/  one row per target sequence (complexes carry several)
  data/parquet/status_history/    one row per status event (status_raw only; 03 adds canon)
  data/parquet/trials/            one row per experimental trial
  data/parquet/protocols/         one row per protocol (free text; host/tag mined in Phase 2)
  data/parquet/outcomes/          one row per PDB deposition (targetpdb_info.csv + PDB databaseRefs)
  data/parquet/targets.fasta      one record per distinct protein seq_md5, for MMseqs2

Usage
  .venv/bin/python scripts/02_parse_tt_xml.py                 # parallel, all centres
  .venv/bin/python scripts/02_parse_tt_xml.py --single        # stream tt.xml.gz
  .venv/bin/python scripts/02_parse_tt_xml.py --centres JCSG MCSG --workers 2
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "TargetTrack" / "TargetTrack-1Jul2017"
BY_CENTRE = RAW / "TargetsbyContributor"
TT_XML = RAW / "TargetTrack XML files" / "tt.xml.gz"
PDB_CSV = RAW / "TargetTrack CSV files" / "targetpdb_info.csv"
CONTRIB_CSV = RAW / "Documentation" / "TargetTrack-Contributors-List.csv"
OUT = ROOT / "data" / "parquet"

ROW_GROUP = 50_000
TABLES = ("targets", "target_sequences", "status_history", "trials", "protocols")

# ---------------------------------------------------------------------------
# ELEMENT MAP (verified against targetTrack-v1.4.1.xsd, 2026-09-07)
#
# targetTrack
#   formatVersion, site, contactInfoList/contactInfo, protocolList/protocol, target*
# protocol[@id="CENTRE-localId"]
#   protocolId, protocolName?, protocolDescription?, protocolType, protocolText?
# target[@id="CENTRE-localId"]
#   targetId, targetName?, dateCreated, dateUpdated, laboratoryList/lab+,
#   contactInfoRefList, projectList/project(projectName, projectId)+,
#   targetRationale, targetAnnotation*, targetCategoryList/targetCategory/targetCategoryName+,
#   targetPartnershipList?, targetProteinType*, status?, targetStopDetails(stopStatus, remark?)?,
#   relatedTargetIdList?, targetSequenceList/(targetSequence+ | targetReference+),
#   ligandList?, url*, remark*, databaseRefList/databaseRef(databaseName, databaseId)+,
#   trialList/trial+
# targetSequence[@id]
#   oneLetterCode, sequenceType, sequenceChemicalType, sequenceConstructType+,
#   sourceOrganism(scientificName, taxDB, taxId), sequenceName, url?, remark*, databaseRefList?
# trial[@id]
#   trialId?, dateUpdated, contactInfoRefList, status, statusHistoryList/statusHistory+,
#   stopDetails(stopStatus, remark?)?, trialSequenceList/trialSequence+,
#   trialProtocolList/protocolRef(protocolId, protocolType, protocolDetails?)+,
#   trialMeasurementList?, trialOutcomeList/trialOutcome(protocolId, outcomeDetails)*
# statusHistory[@id]
#   lab, status, dateComplete, stepDurationDays?, remark?, prevStatusHistoryId?,
#   trialProtocolId*, trialMeasurementId*, trialOutcomeId*
# trialSequence[@id]
#   oneLetterCode, sequenceChemicalType?, sequenceConstructType*, sequenceModifications?,
#   sequenceDetails?, databaseRefList?
#
# Centre acronym: the prefix of target/@id before the first "-" (e.g. "JCSG-419184").
# Status vocabulary: the <status> enumeration in the XSD (30 values); canonicalised in 03.
# ---------------------------------------------------------------------------

SCHEMAS = {
    "targets": pa.schema([
        ("target_id", pa.string()),        # "JCSG-419184": the XML @id, unique archive-wide
        ("local_id", pa.string()),         # <targetId>, unique within a centre
        ("centre", pa.string()),
        ("source_file", pa.string()),
        ("target_name", pa.string()),
        ("date_created", pa.string()),     # ISO date strings; DuckDB casts on read
        ("date_updated", pa.string()),
        ("labs", pa.string()),             # "|"-joined
        ("projects", pa.string()),         # "|"-joined projectName:projectId
        ("categories", pa.string()),       # "|"-joined targetCategoryName
        ("protein_types", pa.string()),    # "|"-joined targetProteinType
        ("partnerships", pa.string()),     # "|"-joined partnershipName
        ("status_raw", pa.string()),       # target-level <status>
        ("stop_status", pa.string()),      # targetStopDetails/stopStatus
        ("stop_remark", pa.string()),
        ("rationale", pa.string()),
        ("annotation", pa.string()),
        ("remark", pa.string()),
        ("n_sequences", pa.int32()),
        ("n_references", pa.int32()),      # targetReference count (complex components)
        ("sequence", pa.string()),         # primary sequence: the longest protein targetSequence
        ("seq_md5", pa.string()),
        ("seq_len", pa.int32()),
        ("sequence_type", pa.string()),
        ("chem_type", pa.string()),
        ("construct_type", pa.string()),   # "|"-joined
        ("organism", pa.string()),
        ("tax_db", pa.string()),
        ("taxon_id", pa.string()),
        ("sequence_name", pa.string()),
        ("database_refs", pa.string()),    # "|"-joined name:id (target + primary sequence)
        ("pdb_ids", pa.string()),          # "|"-joined PDB ids found in databaseRefs
        ("n_trials", pa.int32()),
        ("n_status_events", pa.int32()),
        ("first_seen", pa.string()),       # min(dateCreated, status dates)
        ("last_seen", pa.string()),        # max(dateUpdated, trial dateUpdated, status dates)
    ]),
    "target_sequences": pa.schema([
        ("target_id", pa.string()),
        ("seq_idx", pa.int64()),           # targetSequence/@id
        ("is_primary", pa.bool_()),
        ("sequence", pa.string()),
        ("seq_md5", pa.string()),
        ("seq_len", pa.int32()),
        ("sequence_type", pa.string()),
        ("chem_type", pa.string()),
        ("construct_type", pa.string()),
        ("organism", pa.string()),
        ("tax_db", pa.string()),
        ("taxon_id", pa.string()),
        ("sequence_name", pa.string()),
        ("database_refs", pa.string()),
    ]),
    "status_history": pa.schema([
        ("target_id", pa.string()),
        ("centre", pa.string()),
        ("trial_id", pa.int64()),
        ("history_id", pa.int64()),
        ("lab", pa.string()),
        ("status_raw", pa.string()),
        ("status_date", pa.string()),
        ("step_duration_days", pa.int64()),
        ("prev_history_id", pa.int64()),
        ("remark", pa.string()),
    ]),
    "trials": pa.schema([
        ("target_id", pa.string()),
        ("centre", pa.string()),
        ("trial_id", pa.int64()),          # trial/@id
        ("trial_local_id", pa.string()),   # <trialId> if present
        ("date_updated", pa.string()),
        ("status_raw", pa.string()),       # trial-level current status
        ("stop_status", pa.string()),
        ("stop_remark", pa.string()),
        ("protocol_refs", pa.string()),    # "|"-joined CENTRE-protocolId
        ("protocol_types", pa.string()),   # "|"-joined, order preserved
        ("protocol_details", pa.string()),
        ("n_sequences", pa.int32()),
        ("sequence", pa.string()),         # the PROTEIN construct actually made (see pick_trial_sequences)
        ("seq_md5", pa.string()),
        ("seq_len", pa.int32()),
        ("sequence_dna", pa.string()),     # the gene, where the centre listed one (codon optimisation)
        ("seq_len_dna", pa.int32()),
        ("chem_type", pa.string()),
        ("construct_type", pa.string()),
        ("sequence_modifications", pa.string()),
        ("sequence_details", pa.string()),
        ("outcome_details", pa.string()),  # "|"-joined trialOutcome text
        ("n_status_events", pa.int32()),
        ("first_status_date", pa.string()),
        ("last_status_date", pa.string()),
        ("free_text_notes", pa.string()),  # remarks + stop remark + details, for Phase 2 mining
    ]),
    "protocols": pa.schema([
        ("protocol_id", pa.string()),      # "CENTRE-localId" (protocol/@id)
        ("centre", pa.string()),
        ("local_id", pa.string()),
        ("name", pa.string()),
        ("description", pa.string()),
        ("protocol_type", pa.string()),
        ("text", pa.string()),
    ]),
    "outcomes": pa.schema([
        ("target_id", pa.string()),
        ("centre", pa.string()),
        ("pdb_id", pa.string()),
        ("method", pa.string()),
        ("title", pa.string()),
        ("deposit_date", pa.string()),
        ("release_date", pa.string()),
        ("pdb_status", pa.string()),
        ("resolution", pa.float32()),      # filled by scripts/fetch_pdb_metadata.py
        ("source", pa.string()),           # targetpdb_info.csv | databaseRef
    ]),
}

_WS = re.compile(r"\s+")
_PDB_ID = re.compile(r"^[1-9][A-Za-z0-9]{3}$")


def text(el, tag: str) -> str:
    """Text of the first direct child `tag`, whitespace-collapsed, '' if absent."""
    c = el.find(tag)
    if c is None or c.text is None:
        return ""
    return _WS.sub(" ", c.text).strip()


def texts(el, path: str) -> list[str]:
    return [_WS.sub(" ", c.text).strip() for c in el.iterfind(path) if c.text and c.text.strip()]


def join(vals) -> str:
    return "|".join(v for v in vals if v)


def clean_seq(raw: str) -> str:
    return _WS.sub("", raw or "").upper()


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest() if s else ""


def to_int(s: str):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def db_refs(el) -> list[tuple[str, str]]:
    out = []
    for ref in el.iterfind("databaseRefList/databaseRef"):
        out.append((text(ref, "databaseName"), text(ref, "databaseId")))
    return out


def load_centres() -> set[str]:
    """Centre acronyms from the contributors list plus the per-centre file names."""
    acr = {p.name.split(".")[0] for p in BY_CENTRE.glob("*.xml.gz") if not p.name.startswith("._")}
    if CONTRIB_CSV.exists():
        with CONTRIB_CSV.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.reader(fh):
                if len(row) >= 4 and row[0] and row[0] != "Acronym" and row[2]:
                    acr.add(row[0])
    return acr


def centre_of(target_xml_id: str, known: set[str]) -> str:
    head = target_xml_id.split("-", 1)[0]
    if head in known:
        return head
    # a centre acronym containing "-" would break the split; try the longest known prefix
    for c in sorted(known, key=len, reverse=True):
        if target_xml_id.startswith(c + "-"):
            return c
    return head


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_protocol(el, known: set[str]) -> dict:
    pid = el.get("id") or ""
    return {
        "protocol_id": pid,
        "centre": centre_of(pid, known),
        "local_id": text(el, "protocolId"),
        "name": text(el, "protocolName"),
        "description": text(el, "protocolDescription"),
        "protocol_type": text(el, "protocolType"),
        "text": text(el, "protocolText"),
    }


def extract_target(el, known: set[str], source_file: str) -> tuple[dict, list, list, list]:
    tid = el.get("id") or ""
    centre = centre_of(tid, known)

    # --- sequences -------------------------------------------------------
    seq_rows = []
    for s in el.iterfind("targetSequenceList/targetSequence"):
        seq = clean_seq(s.findtext("oneLetterCode"))
        org = s.find("sourceOrganism")
        refs = db_refs(s)
        seq_rows.append({
            "target_id": tid,
            "seq_idx": to_int(s.get("id")),
            "is_primary": False,
            "sequence": seq,
            "seq_md5": md5(seq),
            "seq_len": len(seq),
            "sequence_type": text(s, "sequenceType"),
            "chem_type": text(s, "sequenceChemicalType"),
            "construct_type": join(texts(s, "sequenceConstructType")),
            "organism": text(org, "scientificName") if org is not None else "",
            "tax_db": text(org, "taxDB") if org is not None else "",
            "taxon_id": text(org, "taxId") if org is not None else "",
            "sequence_name": text(s, "sequenceName"),
            "database_refs": join(f"{n}:{i}" for n, i in refs),
        })
    n_refs = sum(1 for _ in el.iterfind("targetSequenceList/targetReference"))

    # primary = longest protein sequence, else longest of any chemistry
    primary = None
    prot = [r for r in seq_rows if r["chem_type"] == "protein" and r["seq_len"] > 0]
    pool = prot or [r for r in seq_rows if r["seq_len"] > 0] or seq_rows
    if pool:
        primary = max(pool, key=lambda r: r["seq_len"])
        primary["is_primary"] = True

    # --- trials, status history -------------------------------------------
    trial_rows, hist_rows = [], []
    all_dates = []
    for t in el.iterfind("trialList/trial"):
        trial_id = to_int(t.get("id"))
        dates = []
        n_ev = 0
        for h in t.iterfind("statusHistoryList/statusHistory"):
            d = text(h, "dateComplete")
            if d:
                dates.append(d)
            n_ev += 1
            hist_rows.append({
                "target_id": tid,
                "centre": centre,
                "trial_id": trial_id,
                "history_id": to_int(h.get("id")),
                "lab": text(h, "lab"),
                "status_raw": text(h, "status"),
                "status_date": d,
                "step_duration_days": to_int(text(h, "stepDurationDays")),
                "prev_history_id": to_int(text(h, "prevStatusHistoryId")),
                "remark": text(h, "remark"),
            })
        all_dates.extend(dates)
        tseqs = list(t.iterfind("trialSequenceList/trialSequence"))
        first, tseq, tseq_dna = pick_trial_sequences(tseqs)
        prefs = list(t.iterfind("trialProtocolList/protocolRef"))
        stop = t.find("stopDetails")
        outcome = texts(t, "trialOutcomeList/trialOutcome/outcomeDetails")
        seq_mod = next((text(e, "sequenceModifications") for e in tseqs if text(e, "sequenceModifications")), "")
        seq_det = next((text(e, "sequenceDetails") for e in tseqs if text(e, "sequenceDetails")), "")
        stop_remark = text(stop, "remark") if stop is not None else ""
        notes = join([seq_mod, seq_det, stop_remark] + outcome
                     + [r for r in texts(t, "statusHistoryList/statusHistory/remark")])
        trial_rows.append({
            "target_id": tid,
            "centre": centre,
            "trial_id": trial_id,
            "trial_local_id": text(t, "trialId"),
            "date_updated": text(t, "dateUpdated"),
            "status_raw": text(t, "status"),
            "stop_status": text(stop, "stopStatus") if stop is not None else "",
            "stop_remark": stop_remark,
            "protocol_refs": join(f"{centre}-{text(p, 'protocolId')}" for p in prefs if text(p, "protocolId")),
            "protocol_types": join(text(p, "protocolType") for p in prefs),
            "protocol_details": join(text(p, "protocolDetails") for p in prefs),
            "n_sequences": len(tseqs),
            "sequence": tseq,
            "seq_md5": md5(tseq),
            "seq_len": len(tseq),
            "sequence_dna": tseq_dna,
            "seq_len_dna": len(tseq_dna),
            "chem_type": text(first, "sequenceChemicalType") if first is not None else "",
            "construct_type": join(texts(first, "sequenceConstructType")) if first is not None else "",
            "sequence_modifications": seq_mod,
            "sequence_details": seq_det,
            "outcome_details": join(outcome),
            "n_status_events": n_ev,
            "first_status_date": min(dates) if dates else "",
            "last_status_date": max(dates) if dates else "",
            "free_text_notes": notes,
        })
        if text(t, "dateUpdated"):
            all_dates.append(text(t, "dateUpdated"))

    # --- target row ----------------------------------------------------------
    refs = db_refs(el) + (db_refs_from_row(primary) if primary else [])
    pdb_ids = sorted({i.upper() for n, i in refs if n.strip().upper() == "PDB" and _PDB_ID.match(i.strip())})
    stop = el.find("targetStopDetails")
    dc, du = text(el, "dateCreated"), text(el, "dateUpdated")
    seen = [d for d in all_dates + [dc, du] if d]
    row = {
        "target_id": tid,
        "local_id": text(el, "targetId"),
        "centre": centre,
        "source_file": source_file,
        "target_name": text(el, "targetName"),
        "date_created": dc,
        "date_updated": du,
        "labs": join(texts(el, "laboratoryList/lab")),
        "projects": join(f"{text(p, 'projectName')}:{text(p, 'projectId')}" for p in el.iterfind("projectList/project")),
        "categories": join(texts(el, "targetCategoryList/targetCategory/targetCategoryName")),
        "protein_types": join(texts(el, "targetProteinType")),
        "partnerships": join(texts(el, "targetPartnershipList/targetPartnership/partnershipName")),
        "status_raw": text(el, "status"),
        "stop_status": text(stop, "stopStatus") if stop is not None else "",
        "stop_remark": text(stop, "remark") if stop is not None else "",
        "rationale": text(el, "targetRationale"),
        "annotation": join(texts(el, "targetAnnotation")),
        "remark": join(texts(el, "remark")),
        "n_sequences": len(seq_rows),
        "n_references": n_refs,
        "sequence": primary["sequence"] if primary else "",
        "seq_md5": primary["seq_md5"] if primary else "",
        "seq_len": primary["seq_len"] if primary else 0,
        "sequence_type": primary["sequence_type"] if primary else "",
        "chem_type": primary["chem_type"] if primary else "",
        "construct_type": primary["construct_type"] if primary else "",
        "organism": primary["organism"] if primary else "",
        "tax_db": primary["tax_db"] if primary else "",
        "taxon_id": primary["taxon_id"] if primary else "",
        "sequence_name": primary["sequence_name"] if primary else "",
        "database_refs": join(f"{n}:{i}" for n, i in refs),
        "pdb_ids": join(pdb_ids),
        "n_trials": len(trial_rows),
        "n_status_events": len(hist_rows),
        "first_seen": min(seen) if seen else "",
        "last_seen": max(seen) if seen else "",
    }
    return row, seq_rows, hist_rows, trial_rows


_NT = set("ACGTUN")


def is_nucleotide(seq: str) -> bool:
    """True if the string is (almost) pure ACGTUN, i.e. a gene rather than a protein.

    Whole centres (NYCOMPS, MPP, NatPro, TMPC, TEMIMPS at 100%, CESG 86%, SGX 87%,
    NYSGXRC 64%) list the DNA construct first in trialSequenceList, and 16,931 of those
    rows carry no sequenceChemicalType element at all (it is minOccurs="0"), so the
    recorded label cannot be trusted on its own. Composition can.
    """
    if len(seq) < 12:
        return False
    return sum(ch in _NT for ch in seq) / len(seq) > 0.95


def pick_trial_sequences(tseqs):
    """(protein_element, protein_seq, dna_seq) from a trial's sequence list.

    Prefers an element recorded as protein whose composition agrees; falls back to the
    first non-nucleotide sequence; finally to the first element, so nothing is dropped.
    """
    parsed = [(el, clean_seq(el.findtext("oneLetterCode"))) for el in tseqs]
    prot = [(el, sq) for el, sq in parsed if sq and not is_nucleotide(sq)]
    dna = [(el, sq) for el, sq in parsed if sq and is_nucleotide(sq)]
    labelled = [(el, sq) for el, sq in prot if text(el, "sequenceChemicalType") == "protein"]
    pick = (labelled or prot or parsed or [(None, "")])[0]
    return pick[0], pick[1], (dna[0][1] if dna else "")


def db_refs_from_row(seq_row: dict) -> list[tuple[str, str]]:
    out = []
    for item in (seq_row.get("database_refs") or "").split("|"):
        if ":" in item:
            n, i = item.split(":", 1)
            out.append((n, i))
    return out


# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------

class PartWriter:
    """Buffers rows per table and writes 50k-row groups to one Parquet part per table."""

    def __init__(self, part: str):
        self.part = part
        self.buf = {t: [] for t in TABLES}
        self.writers = {}
        self.counts = Counter()
        for t in TABLES:
            (OUT / t).mkdir(parents=True, exist_ok=True)

    def add(self, table: str, rows):
        b = self.buf[table]
        b.extend(rows)
        self.counts[table] += len(rows)
        if len(b) >= ROW_GROUP:
            self.flush(table)

    def flush(self, table: str):
        rows = self.buf[table]
        if not rows:
            return
        schema = SCHEMAS[table]
        tbl = pa.Table.from_pylist(rows, schema=schema)
        if table not in self.writers:
            self.writers[table] = pq.ParquetWriter(OUT / table / f"{self.part}.parquet", schema, compression="zstd")
        self.writers[table].write_table(tbl, row_group_size=ROW_GROUP)
        self.buf[table] = []

    def close(self):
        for t in TABLES:
            self.flush(t)
            if t not in self.writers:  # write an empty part so DuckDB sees a consistent schema
                pq.write_table(SCHEMAS[t].empty_table(), OUT / t / f"{self.part}.parquet")
        for w in self.writers.values():
            w.close()


def parse_file(path: Path, part: str) -> dict:
    """Stream one TargetTrack XML file into Parquet parts. Returns counts + vocab tallies."""
    known = load_centres()
    t0 = time.time()
    w = PartWriter(part)
    status_vocab = Counter()
    fasta = {}  # seq_md5 -> sequence (protein only)
    with gzip.open(path, "rb") as fh:
        # recover=True: the archive is ISO-8859-1 with occasional stray bytes
        parser_kwargs = dict(events=("end",), tag=("target", "protocol"), recover=True, huge_tree=True)
        for _, el in etree.iterparse(fh, **parser_kwargs):
            if el.tag == "protocol":
                w.add("protocols", [extract_protocol(el, known)])
            else:
                row, seqs, hist, trials = extract_target(el, known, path.name)
                w.add("targets", [row])
                w.add("target_sequences", seqs)
                w.add("status_history", hist)
                w.add("trials", trials)
                for h in hist:
                    status_vocab[h["status_raw"]] += 1
                if row["chem_type"] == "protein" and row["seq_md5"]:
                    fasta.setdefault(row["seq_md5"], row["sequence"])
            # memory-safe clearing pattern (spec section 4.2)
            el.clear()
            while el.getprevious() is not None:
                del el.getparent()[0]
    w.close()
    fa = OUT / "fasta_parts" / f"{part}.fasta"
    fa.parent.mkdir(parents=True, exist_ok=True)
    with fa.open("w") as out:
        for k, s in fasta.items():
            out.write(f">{k}\n{s}\n")
    return {"part": part, "seconds": round(time.time() - t0, 1),
            "counts": dict(w.counts), "status_vocab": dict(status_vocab)}


# ---------------------------------------------------------------------------
# Outcomes from targetpdb_info.csv (+ PDB databaseRefs, added after the parse)
# ---------------------------------------------------------------------------

def write_outcomes(known: set[str]) -> int:
    (OUT / "outcomes").mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    with PDB_CSV.open(encoding="latin-1", newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for r in rdr:
            tid = (r.get("tt_id") or "").strip()
            pdb = (r.get("pdb_id") or "").strip().upper()
            if not tid or not pdb:
                continue
            seen.add((tid, pdb))
            rows.append({
                "target_id": tid, "centre": centre_of(tid, known), "pdb_id": pdb,
                "method": (r.get("method") or "").strip(), "title": (r.get("title") or "").strip(),
                "deposit_date": (r.get("date_deposited") or "").strip(),
                "release_date": (r.get("date_released") or "").strip(),
                "pdb_status": (r.get("pdb_status") or "").strip(),
                "resolution": None, "source": "targetpdb_info.csv",
            })
    # PDB ids that only appear as databaseRefs inside target records
    import duckdb  # local import: keep the worker processes light
    q = duckdb.query(f"SELECT target_id, centre, pdb_ids FROM read_parquet('{OUT / 'targets' / '*.parquet'}') WHERE pdb_ids <> ''")
    for tid, centre, ids in q.fetchall():
        for pdb in ids.split("|"):
            if (tid, pdb) not in seen:
                seen.add((tid, pdb))
                rows.append({"target_id": tid, "centre": centre, "pdb_id": pdb, "method": "", "title": "",
                             "deposit_date": "", "release_date": "", "pdb_status": "", "resolution": None,
                             "source": "databaseRef"})
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS["outcomes"]), OUT / "outcomes" / "outcomes.parquet",
                   compression="zstd")
    return len(rows)


def merge_fasta() -> int:
    seen, n = set(), 0
    with (OUT / "targets.fasta").open("w") as out:
        for p in sorted((OUT / "fasta_parts").glob("*.fasta")):
            with p.open() as fh:
                hdr = None
                for line in fh:
                    if line.startswith(">"):
                        hdr = line[1:].strip()
                    elif hdr and hdr not in seen:
                        seen.add(hdr)
                        out.write(f">{hdr}\n{line}")
                        n += 1
    shutil.rmtree(OUT / "fasta_parts")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single", action="store_true", help="stream tt.xml.gz in one process instead of the per-centre files")
    ap.add_argument("--centres", nargs="*", help="subset of centre acronyms (per-centre mode)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--keep", action="store_true", help="do not wipe data/parquet first")
    args = ap.parse_args()

    if not args.keep and OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.single:
        jobs = [(TT_XML, "tt")]
    else:
        files = sorted(p for p in BY_CENTRE.glob("*.xml.gz") if not p.name.startswith("._"))  # skip AppleDouble forks
        if args.centres:
            files = [f for f in files if f.name.split(".")[0] in set(args.centres)]
        jobs = [(f, f.name.split(".")[0]) for f in files]
    if not jobs:
        sys.exit("no input files found; run scripts/01_fetch_zenodo.py first")

    print(f"parsing {len(jobs)} file(s) with {1 if args.single else args.workers} worker(s) -> {OUT}")
    t0 = time.time()
    results = []
    if args.single or args.workers == 1:
        for path, part in jobs:
            r = parse_file(path, part)
            print(f"  {part:<14} {r['counts'].get('targets', 0):>8,} targets  {r['seconds']:>7.1f}s")
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(parse_file, path, part): part for path, part in jobs}
            for f in as_completed(futs):
                r = f.result()
                print(f"  {r['part']:<14} {r['counts'].get('targets', 0):>8,} targets  {r['seconds']:>7.1f}s", flush=True)
                results.append(r)

    totals, vocab = Counter(), Counter()
    for r in results:
        totals.update(r["counts"])
        vocab.update(r["status_vocab"])
    n_fa = merge_fasta()
    n_out = write_outcomes(load_centres())

    summary = {
        "inputs": [str(p) for p, _ in jobs],
        "seconds": round(time.time() - t0, 1),
        "rows": dict(totals) | {"outcomes": n_out, "fasta_records": n_fa},
        "status_vocab": dict(sorted(vocab.items(), key=lambda kv: -kv[1])),
    }
    (OUT / "parse_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ndone in {summary['seconds']}s")
    for k, v in summary["rows"].items():
        print(f"  {k:<18} {v:>10,}")
    print(f"  distinct status_raw values in status_history: {len(vocab)}")


if __name__ == "__main__":
    main()
