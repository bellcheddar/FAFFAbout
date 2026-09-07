#!/usr/bin/env python
"""05_derive_features.py: the feature table, in four parts.

  features_sequence  one row per distinct seq_md5   composition, topology, disorder
  features_target    one row per target             taxonomy, annotation, construct, host
  features_context   one row per (target, gate)     archive context, leak-safe
  features           the join, one row per target   what the GBM and the app consume

The raw sequence NEVER goes into a prompt (spec section 5.4). These numbers do.

Leakage is the thing to get right here. Archive-context features describe the precedents
a scientist could have looked up, so they are computed over the TRAINING split only, and
for a training target its own outcome is subtracted back out. Without that leave-one-out
a target reads its own fate off its cluster's base rate, and every metric flatters itself.

Usage
  .venv/bin/python scripts/05_derive_features.py                  # everything
  .venv/bin/python scripts/05_derive_features.py --skip-sequence  # reuse cached seq features
"""
from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import features_seq as fs          # noqa: E402
from taxonomy import Taxonomy      # noqa: E402

DB = ROOT / "data" / "faffabout.duckdb"
PQ = ROOT / "data" / "parquet"
OUT = PQ / "features"
N_GATES = 8

# --------------------------------------------------------------------------- text mining
# Host, tag, protease and induction regime live in 1,501 shared protocol documents that
# 918,689 trials (95.5%) point at, so mining ~1,500 texts annotates almost the whole
# archive. Patterns are ordered: the first match wins.
HOST_PATTERNS = [
    ("cell_free",   r"cell[- ]free|wheat germ|in vitro translation|PURE system"),
    ("insect",      r"\bSf9\b|\bSf21\b|baculovir|High Five|insect cell"),
    ("mammalian",   r"\bHEK ?293|\bCHO\b|mammalian cell"),
    ("yeast",       r"\bPichia\b|\bS\. cerevisiae\b|yeast expression"),
    ("ecoli",       r"\bBL21|\bE\.? ?coli\b|Rosetta|Origami|Tuner|C41|C43|Arctic ?Express|"
                    r"\bDE3\b|\bTOP10\b|\bJM109\b"),
]
TAG_PATTERNS = [
    ("his",      r"his[- ]?tag|hexahistidine|polyhistidine|6 ?x ?his|his[- ]?6|his ?8|HHHHHH"),
    ("gst",      r"\bGST\b|glutathione S-transferase"),
    ("mbp",      r"\bMBP\b|maltose[- ]binding"),
    ("sumo",     r"\bSUMO\b"),
    ("thioredoxin", r"thioredoxin|\bTrxA\b"),
    ("strep",    r"strep[- ]?tag|streptavidin"),
]
PROTEASE_PATTERNS = [
    ("tev",        r"\bTEV\b|tobacco etch"),
    ("thrombin",   r"\bthrombin\b"),
    ("prescission", r"pre ?scission|\bHRV ?3C\b|\b3C protease"),
    ("enterokinase", r"enterokinase|\bEK\b cleav"),
    ("sumo_protease", r"\bUlp1\b|SUMO protease"),
]
OTHER_PATTERNS = {
    "selenomethionine": r"selenomet|\bSeMet\b|Se-Met",
    "autoinduction":    r"auto[- ]?induc",
    "iptg":             r"\bIPTG\b",
    "codon_optimised":  r"codon[- ]optimi|codon usage engineered|synthetic gene|gene synthesis",
    "cold_shock":       r"cold[- ]?shock|Arctic ?Express|\b1[0-8] ?°? ?C\b",
    "refolding":        r"refold|inclusion bod",
    "detergent":        r"\bDDM\b|detergent|\bLDAO\b|\bOG\b solubil|\bCHAPS\b",
}

# Structured fields the centres wrote into the trial free text. Measured coverage:
# boundaries 17.1% of trials (residue-scale, verified), expression level 12.8%,
# solubility 7.1%, final concentration 4.9%.
RE_BOUNDS = re.compile(r"Sequence start:\s*(\d+)\s*Sequence end:\s*(\d+)")
RE_EXPR = re.compile(r"Expression level\s*:\s*(\d+)\s*%")
RE_SOL = re.compile(r"Solubility level\s*:\s*(\d+)\s*%")
RE_CONC = re.compile(r"Final concentration:\s*([0-9.]+)\s*mg/mL")
RE_STEP_FAILED = re.compile(r"step failed")


def match_first(patterns, text: str) -> str:
    low = text or ""
    for name, pat in patterns:
        if re.search(pat, low, re.I):
            return name
    return ""


# --------------------------------------------------------------------------- sequence

def _seq_chunk(rows: list[tuple[str, str]]) -> pd.DataFrame:
    md5s = [m for m, _ in rows]
    seqs = [s for _, s in rows]
    counts = fs.counts_matrix(seqs)
    out = {
        "seq_md5": md5s,
        "seq_len": np.array([len(s) for s in seqs], dtype=np.int32),
        "pi": fs.isoelectric_point(counts),
        "gravy": fs.gravy(counts),
        "net_charge_ph7": fs.net_charge(counts, 7.0),
        "aromatic_frac": fs.aromatic_fraction(counts),
        "cys_frac": fs.residue_fraction(counts, "C"),
        "met_frac": fs.residue_fraction(counts, "M"),
        "gly_frac": fs.residue_fraction(counts, "G"),
        "pro_frac": fs.residue_fraction(counts, "P"),
        "cys_count": counts[:, fs.AA_INDEX["C"]].astype(np.int32),
        "met_count": counts[:, fs.AA_INDEX["M"]].astype(np.int32),
    }
    tm, sig, dis, dn, dc, di, lc = [], [], [], [], [], [], []
    for s in seqs:
        tm.append(fs.tm_helices(s))
        sig.append(fs.has_signal_peptide(s))
        a, b, c, d = fs.disorder_fractions(s)
        dis.append(a); dn.append(b); dc.append(c); di.append(d)
        lc.append(fs.low_complexity_fraction(s))
    out |= {
        "tm_helices": np.array(tm, dtype=np.int16),
        "signal_peptide": np.array(sig, dtype=bool),
        "disorder_frac": np.array(dis), "disorder_nterm": np.array(dn),
        "disorder_cterm": np.array(dc), "disorder_internal": np.array(di),
        "low_complexity_frac": np.array(lc),
    }
    return pd.DataFrame(out)


def build_sequence_features(con: duckdb.DuckDBPyConnection, workers: int) -> None:
    rows = con.execute("""
        SELECT DISTINCT seq_md5, sequence FROM targets
        WHERE seq_md5 <> '' AND sequence <> '' AND chem_type = 'protein'
    """).fetchall()
    print(f"sequence features for {len(rows):,} distinct sequences on {workers} workers ...")
    chunks = [rows[i:i + 5000] for i in range(0, len(rows), 5000)]
    frames = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, df in enumerate(ex.map(_seq_chunk, chunks), 1):
            frames.append(df)
            print(f"  {min(i*5000, len(rows)):,}/{len(rows):,}", end="\r", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(OUT / "features_sequence.parquet", index=False)
    print(f"\nwrote {OUT / 'features_sequence.parquet'}")


# --------------------------------------------------------------------------- target level

def build_target_features(con: duckdb.DuckDBPyConnection) -> None:
    print("taxonomy ...")
    tx = Taxonomy()
    tgt = con.execute("SELECT target_id, organism, taxon_id FROM targets").df()
    cls = [tx.classify(t, o) for t, o in zip(tgt.taxon_id, tgt.organism)]
    tax = pd.DataFrame(cls)
    tax.insert(0, "target_id", tgt.target_id.values)
    tax.to_parquet(OUT / "_tax.parquet", index=False)
    print(f"  resolved {(tax.resolved_by != '').mean()*100:.1f}% of targets "
          f"({(tax.resolved_by == 'id').mean()*100:.1f}% by id, "
          f"{(tax.resolved_by == 'name').mean()*100:.1f}% by name)")

    print("mining protocol text for host, tag and protease ...")
    prot = con.execute("""
        SELECT protocol_id, coalesce(name,'') || ' ' || coalesce(description,'') || ' '
               || coalesce(text,'') AS blob FROM protocols
    """).df()
    prot["host"] = [match_first(HOST_PATTERNS, b) for b in prot.blob]
    prot["tag"] = [match_first(TAG_PATTERNS, b) for b in prot.blob]
    prot["protease"] = [match_first(PROTEASE_PATTERNS, b) for b in prot.blob]
    for k, pat in OTHER_PATTERNS.items():
        prot[k] = [bool(re.search(pat, b, re.I)) for b in prot.blob]
    prot.drop(columns=["blob"]).to_parquet(OUT / "_protocol_terms.parquet", index=False)
    print(f"  {len(prot):,} protocols: host {(prot.host != '').mean()*100:.0f}%, "
          f"tag {(prot.tag != '').mean()*100:.0f}%, protease {(prot.protease != '').mean()*100:.0f}%")

    print("trial free text ...")
    con.execute(f"CREATE OR REPLACE VIEW protocol_terms AS SELECT * FROM read_parquet('{OUT / '_protocol_terms.parquet'}')")
    con.execute(f"""
        COPY (
            WITH tp AS (
                SELECT t.target_id, t.trial_id,
                       unnest(string_split(t.protocol_refs, '|')) AS protocol_id
                FROM trials t WHERE t.protocol_refs <> ''
            ),
            per_trial AS (
                SELECT tp.target_id, tp.trial_id,
                       max(NULLIF(pt.host, ''))     AS host,
                       max(NULLIF(pt.tag, ''))      AS tag,
                       max(NULLIF(pt.protease, '')) AS protease,
                       bool_or(pt.selenomethionine) AS selenomethionine,
                       bool_or(pt.autoinduction)    AS autoinduction,
                       bool_or(pt.codon_optimised)  AS codon_optimised,
                       bool_or(pt.refolding)        AS refolding,
                       bool_or(pt.detergent)        AS detergent
                FROM tp JOIN protocol_terms pt USING (protocol_id)
                GROUP BY 1, 2
            )
            SELECT t.target_id,
                   count(*)                                   AS n_trials,
                   max(NULLIF(p.host, ''))                    AS host,
                   max(NULLIF(p.tag, ''))                     AS tag,
                   max(NULLIF(p.protease, ''))                AS protease,
                   coalesce(bool_or(p.selenomethionine), false) AS selenomethionine,
                   coalesce(bool_or(p.autoinduction), false)    AS autoinduction,
                   coalesce(bool_or(p.codon_optimised), false)  AS codon_optimised,
                   coalesce(bool_or(p.refolding), false)        AS refolding,
                   coalesce(bool_or(p.detergent), false)        AS detergent,
                   max(NULLIF(t.construct_type, ''))          AS construct_type,
                   count(*) FILTER (WHERE t.sequence_dna <> '') AS n_trials_with_gene,
                   -- structured fields the centres wrote into the free text
                   max(try_cast(regexp_extract(t.free_text_notes, 'Sequence start:\\s*(\\d+)', 1) AS INTEGER))  AS construct_start,
                   max(try_cast(regexp_extract(t.free_text_notes, 'Sequence end:\\s*(\\d+)', 1) AS INTEGER))    AS construct_end,
                   max(try_cast(regexp_extract(t.free_text_notes, 'Expression level\\s*:\\s*(\\d+)', 1) AS INTEGER)) AS expression_pct,
                   max(try_cast(regexp_extract(t.free_text_notes, 'Solubility level\\s*:\\s*(\\d+)', 1) AS INTEGER))  AS solubility_pct,
                   max(try_cast(regexp_extract(t.free_text_notes, 'Final concentration:\\s*([0-9.]+)', 1) AS DOUBLE)) AS final_conc_mg_ml,
                   bool_or(t.free_text_notes LIKE '%step failed%')  AS any_step_failed,
                   max(t.seq_len)                             AS max_trial_seq_len,
                   min(NULLIF(t.seq_len, 0))                  AS min_trial_seq_len
            FROM trials t LEFT JOIN per_trial p USING (target_id, trial_id)
            GROUP BY 1
        ) TO '{OUT / '_trial_terms.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    con.execute(f"""
        COPY (
            SELECT t.target_id, t.centre, t.organism, t.seq_md5, t.seq_len,
                   t.protein_types, t.categories, t.construct_type AS target_construct_type,
                   t.sequence_type, t.n_sequences, t.n_references,
                   t.database_refs LIKE '%PFAM%' OR t.database_refs LIKE '%Pfam%' AS has_pfam,
                   lower(t.database_refs) LIKE '%uniprot%'                        AS has_uniprot,
                   t.pdb_ids <> ''                                                AS has_pdb_ref,
                   t.protein_types LIKE '%membrane protein%'                      AS archive_says_membrane,
                   t.protein_types LIKE '%eukaryotic protein%'                    AS archive_says_eukaryotic,
                   t.protein_types LIKE '%multidomain%'                           AS archive_says_multidomain,
                   x.superkingdom, x.kingdom, x.phylum, x.genus, x.resolved_by AS tax_resolved_by,
                   r.* EXCLUDE (target_id)
            FROM targets t
            LEFT JOIN read_parquet('{OUT / '_tax.parquet'}') x USING (target_id)
            LEFT JOIN read_parquet('{OUT / '_trial_terms.parquet'}') r USING (target_id)
        ) TO '{OUT / 'features_target.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"wrote {OUT / 'features_target.parquet'}")


# --------------------------------------------------------------------------- archive context

def build_context_features(con: duckdb.DuckDBPyConnection) -> None:
    """Per (target, gate) precedent counts and base rates, computed over TRAIN only,
    with the target's own contribution subtracted back out."""
    print("archive context (leave-one-out over the training split) ...")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW pool AS
        SELECT s.target_id, s.cluster_id, s.centre, s.max_stage, s.censored,
               s.split_cluster,
               c70.cluster_id AS cluster70
        FROM splits s
        LEFT JOIN read_parquet('{ROOT / 'data' / 'clusters' / 'tt70_clusters.parquet'}') c70
               ON c70.seq_md5 = (SELECT t.seq_md5 FROM targets t WHERE t.target_id = s.target_id)
        WHERE s.cluster_id IS NOT NULL
    """)
    # Cluster totals over the eligible pool: training split, uncensored.
    con.execute("""
        CREATE OR REPLACE TEMP VIEW gate_totals AS
        SELECT p.cluster_id, g.gate,
               count(*) FILTER (WHERE p.max_stage >= g.gate)  AS n_entered,
               count(*) FILTER (WHERE p.max_stage >  g.gate)  AS n_cleared
        FROM pool p CROSS JOIN (SELECT unnest(generate_series(0, 7)) AS gate) g
        WHERE p.split_cluster = 'train' AND NOT p.censored
        GROUP BY 1, 2
    """)
    con.execute(f"""
        COPY (
            SELECT p.target_id, g.gate,
                   -- subtract this target's own contribution when it is part of the pool
                   t.n_entered - CASE WHEN p.split_cluster = 'train' AND NOT p.censored
                                           AND p.max_stage >= g.gate THEN 1 ELSE 0 END AS n_precedents,
                   t.n_cleared - CASE WHEN p.split_cluster = 'train' AND NOT p.censored
                                           AND p.max_stage >  g.gate THEN 1 ELSE 0 END AS n_precedents_cleared
            FROM pool p
            CROSS JOIN (SELECT unnest(generate_series(0, 7)) AS gate) g
            LEFT JOIN gate_totals t ON t.cluster_id = p.cluster_id AND t.gate = g.gate
        ) TO '{OUT / '_ctx_raw.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute(f"""
        COPY (
            SELECT target_id, gate,
                   coalesce(n_precedents, 0)         AS n_precedents,
                   coalesce(n_precedents_cleared, 0) AS n_precedents_cleared,
                   CASE WHEN coalesce(n_precedents, 0) > 0
                        THEN 1.0 * n_precedents_cleared / n_precedents END AS cluster_base_rate
            FROM read_parquet('{OUT / '_ctx_raw.parquet'}')
        ) TO '{OUT / 'features_context.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
    """)
    # Per-target cluster summary, same leave-one-out discipline.
    con.execute(f"""
        COPY (
            WITH tot AS (
                SELECT cluster_id,
                       count(*) FILTER (WHERE split_cluster = 'train')                     AS n_train,
                       count(*) FILTER (WHERE split_cluster = 'train' AND NOT censored)     AS n_train_unc,
                       count(*) FILTER (WHERE split_cluster = 'train' AND censored)         AS n_train_cens,
                       count(*) FILTER (WHERE split_cluster = 'train' AND NOT censored
                                              AND max_stage >= 8)                           AS n_train_dep
                FROM pool GROUP BY 1),
            t70 AS (
                SELECT cluster70,
                       count(*) FILTER (WHERE split_cluster = 'train' AND NOT censored)     AS n70
                FROM pool WHERE cluster70 IS NOT NULL GROUP BY 1)
            SELECT p.target_id, p.cluster_id, p.cluster70,
                   tot.n_train_unc - CASE WHEN p.split_cluster = 'train' AND NOT p.censored
                                          THEN 1 ELSE 0 END                       AS n_cluster_precedents,
                   coalesce(t70.n70, 0) - CASE WHEN p.split_cluster = 'train' AND NOT p.censored
                                               AND p.cluster70 IS NOT NULL THEN 1 ELSE 0 END
                                                                                  AS n_close_precedents,
                   CASE WHEN tot.n_train > 0 THEN 1.0 * tot.n_train_cens / tot.n_train END
                                                                                  AS cluster_censored_frac,
                   CASE WHEN (tot.n_train_unc - CASE WHEN p.split_cluster = 'train'
                                                          AND NOT p.censored THEN 1 ELSE 0 END) > 0
                        THEN 1.0 * (tot.n_train_dep - CASE WHEN p.split_cluster = 'train'
                                                                AND NOT p.censored AND p.max_stage >= 8
                                                           THEN 1 ELSE 0 END)
                             / (tot.n_train_unc - CASE WHEN p.split_cluster = 'train'
                                                            AND NOT p.censored THEN 1 ELSE 0 END)
                        END                                                       AS cluster_deposit_rate,
                   -- the identity band a precedent search would report
                   CASE WHEN coalesce(t70.n70, 0) - CASE WHEN p.split_cluster = 'train' AND NOT p.censored
                                                              AND p.cluster70 IS NOT NULL THEN 1 ELSE 0 END > 0
                             THEN 0.70
                        WHEN tot.n_train_unc - CASE WHEN p.split_cluster = 'train' AND NOT p.censored
                                                    THEN 1 ELSE 0 END > 0 THEN 0.30
                        ELSE 0.0 END                                              AS closest_identity_band
            FROM pool p
            LEFT JOIN tot USING (cluster_id)
            LEFT JOIN t70 ON t70.cluster70 = p.cluster70
        ) TO '{OUT / 'features_cluster.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"wrote {OUT / 'features_context.parquet'} and features_cluster.parquet")


def build_join(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
        COPY (
            SELECT ft.*, fs.* EXCLUDE (seq_md5, seq_len), fc.* EXCLUDE (target_id, cluster_id)
            FROM read_parquet('{OUT / 'features_target.parquet'}') ft
            LEFT JOIN read_parquet('{OUT / 'features_sequence.parquet'}') fs USING (seq_md5)
            LEFT JOIN read_parquet('{OUT / 'features_cluster.parquet'}') fc USING (target_id)
        ) TO '{OUT / 'features.parquet'}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
    """)
    for name in ("features", "features_sequence", "features_target", "features_context", "features_cluster"):
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{OUT / (name + '.parquet')}')")
    print(f"wrote {OUT / 'features.parquet'}")


def report(con: duckdb.DuckDBPyConnection) -> None:
    print("\n=== coverage ===")
    print(con.execute("""
        SELECT count(*) AS targets,
               round(100.0*avg(CASE WHEN pi IS NOT NULL THEN 1 ELSE 0 END), 1)          AS pct_sequence,
               round(100.0*avg(CASE WHEN superkingdom <> '' THEN 1 ELSE 0 END), 1)      AS pct_kingdom,
               round(100.0*avg(CASE WHEN host IS NOT NULL THEN 1 ELSE 0 END), 1)        AS pct_host,
               round(100.0*avg(CASE WHEN tag IS NOT NULL THEN 1 ELSE 0 END), 1)         AS pct_tag,
               round(100.0*avg(CASE WHEN construct_start IS NOT NULL THEN 1 ELSE 0 END), 1) AS pct_bounds,
               round(100.0*avg(CASE WHEN expression_pct IS NOT NULL THEN 1 ELSE 0 END), 1)  AS pct_expr,
               round(100.0*avg(CASE WHEN n_cluster_precedents > 0 THEN 1 ELSE 0 END), 1)    AS pct_has_precedent
        FROM features
    """).df().to_string(index=False))
    print("\nkingdom:")
    print(con.execute("""
        SELECT coalesce(nullif(superkingdom,''),'unresolved') AS superkingdom, count(*) AS n,
               round(100.0*count(*)/sum(count(*)) OVER (), 1) AS pct,
               round(avg(seq_len)) AS mean_len, round(avg(tm_helices), 2) AS mean_tm
        FROM features GROUP BY 1 ORDER BY n DESC
    """).df().to_string(index=False))
    print("\nhost (from protocol text):")
    print(con.execute("""
        SELECT coalesce(host,'unknown') AS host, count(*) AS n,
               round(100.0*count(*)/sum(count(*)) OVER (), 1) AS pct
        FROM features GROUP BY 1 ORDER BY n DESC
    """).df().to_string(index=False))
    print("\ndoes the archive agree with the TM prediction?")
    print(con.execute("""
        SELECT archive_says_membrane, round(avg(tm_helices), 2) AS mean_tm_predicted,
               round(100.0*avg(CASE WHEN tm_helices >= 3 THEN 1 ELSE 0 END), 1) AS pct_polytopic,
               count(*) AS n
        FROM features WHERE tm_helices IS NOT NULL GROUP BY 1
    """).df().to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sequence", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))
    con.execute(f"CREATE OR REPLACE VIEW splits AS SELECT * FROM read_parquet('{PQ}/splits/splits.parquet')")
    if not args.skip_sequence:
        build_sequence_features(con, args.workers)
    build_target_features(con)
    build_context_features(con)
    build_join(con)
    report(con)
    con.close()


if __name__ == "__main__":
    main()
