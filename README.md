# 🪦 FAFFAbout

> **Fine-tuned Attrition Forecasting From Archives: know where your protein is likely to die before you order the gene.**

![python](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white) ![lxml](https://img.shields.io/badge/lxml-6.1-467FF7) ![pyarrow](https://img.shields.io/badge/pyarrow-25.0-467FF7) ![duckdb](https://img.shields.io/badge/duckdb-1.5-FFF000?logo=duckdb&logoColor=black) ![pandas](https://img.shields.io/badge/pandas-3.0-150458?logo=pandas&logoColor=white) ![mmseqs2](https://img.shields.io/badge/MMseqs2-18-00897B) ![targets](https://img.shields.io/badge/targets-335%2C771-467FF7) ![status events](https://img.shields.io/badge/status%20events-3.78M-467FF7) ![clusters](https://img.shields.io/badge/clusters%20(30%25%20id)-88%2C452-467FF7) ![tests](https://img.shields.io/badge/pytest-13%20passing-00897B) ![data](https://img.shields.io/badge/data-PSI%20TargetTrack%20%C2%B7%20CC--BY--SA--4.0-9b51e0) ![phase 1](https://img.shields.io/badge/phase%201-complete-fcb900) ![phase 2](https://img.shields.io/badge/phase%202-in%20progress-fcb900) [![MLX-LM](https://img.shields.io/badge/MLX--LM-Apple%20Silicon-000000?logo=apple&logoColor=white)](https://github.com/ml-explore/mlx-lm) ![author](https://img.shields.io/badge/author-Marc%20C.%20Deller%2C%20D.Phil.-1C244B)

<table>
<tr>
<td>🌐 <b>Website</b></td><td><a href="https://marcdeller.com" target="_blank" rel="noopener noreferrer">marcdeller.com</a></td>
<td>✉️ <b>Contact</b></td><td><a href="mailto:marc@marcdeller.com">marc@marcdeller.com</a></td>
<td>🐙 <b>GitHub</b></td><td><a href="https://github.com/bellcheddar/FAFFAbout" target="_blank" rel="noopener noreferrer">bellcheddar/FAFFAbout</a></td>
</tr>
</table>

---

FAFFAbout is a decision-support tool for the first week of a structure project. You paste a FASTA, a UniProt accession or a PDB ID and get a breakdown of the protein plus calibrated guidance on where comparable targets historically stalled (cloning, expression, solubility, purification, crystallisation, diffraction, structure, deposition), which construct and host choices moved the needle, and how much confidence the evidence actually supports. It is grounded in the Protein Structure Initiative's TargetTrack archive: 335,771 targets, 961,548 experimental trials and 3.8 million status events from 41 structural genomics centres, 2000 to 2017.

**Why it matters:** the PSI archive is the only large corpus that records where protein production attempts *stopped*, not just which ones succeeded, and nobody has turned it into a forecasting tool. The trap is that a target parked at "cloned" in June 2017 is one of three things (failed on scientific grounds, censored because its centre's funding ended, or still in flight when the archive froze), and a model that cannot tell them apart learns to predict when US funding programmes ended. FAFFAbout derives censoring empirically per centre, keeps it as a first-class flag through every table, and never lets it become a negative label. It is useful for: triaging a target list before ordering genes, choosing between orthologues, deciding on a host and tag with evidence rather than habit, and setting honest expectations with collaborators.

## 🧭 What it is not

It is not a structure predictor, a fold recogniser or a replacement for AlphaFold. It predicts *experimental attrition*, which is a different and largely unmodelled quantity. The archive has a selection bias too: PSI targets already passed a soluble-prokaryotic filter, so predictions for membrane and eukaryotic proteins are extrapolation, and the interface says so.

## 🧱 Stack

| Layer | Choice | Why |
|---|---|---|
| Archive | Zenodo [10.5281/zenodo.821654](https://doi.org/10.5281/zenodo.821654), TargetTrack 1 July 2017 (795 MiB, CC-BY-SA-4.0) | The complete PSI record, frozen |
| Parse | `lxml.iterparse` streaming, one process per centre file | `tt.xml` is 1.45 GB decompressed; a DOM parse would need tens of GB |
| Store | Parquet (zstd, 50k-row groups) with DuckDB views | Static 336k-row analytical store; group-bys are an order of magnitude faster than SQLite |
| Clustering | MMseqs2 `easy-cluster`, 30% identity, 80% coverage of the shorter sequence | Every split is by cluster, never by row; the same protein appears across five or more orthologues |
| Baseline | LightGBM on tabular features (Phase 3) | The gate the LLM must clear; ships in the app as the source of every probability |
| Model | Llama-3.1-8B-Instruct, 8-bit, LoRA on all 32 layers via MLX-LM (Phase 3) | Writes the narrative only; every number comes from the GBM or DuckDB |
| Serve | Flask + gunicorn + nginx on faffabout.mdeller.com (Phase 4) | Same shape as AlphaFraud |

## 🔧 Installation

```bash
cd FAFFAbout
uv venv --python 3.14 .venv
uv pip install -r requirements.txt
brew install mmseqs2
```

## 🚀 Usage: Phase 1 pipeline

| Step | Command | Output | Time on M1 Max |
|---|---|---|---|
| 1. Fetch | `.venv/bin/python scripts/01_fetch_zenodo.py` | `data/raw/TargetTrack/`, md5-verified, licence recorded in `data/raw/zenodo_record.json` | 5 min download |
| 2. Parse | `.venv/bin/python scripts/02_parse_tt_xml.py --workers 8` | `data/parquet/{targets,target_sequences,status_history,trials,protocols,outcomes}/`, `targets.fasta` | 62 s with 8 workers |
| 3. Normalise | `.venv/bin/python scripts/03_normalise_status.py` | `status_map.parquet`, `status_unmapped.csv`, `status_history_canon/`, `target_stage/` | under 1 min |
| 4. Cluster | `bash scripts/04_cluster_sequences.sh` | `data/clusters/tt30_clusters.parquet` | 2 min, 10 threads |
| 5. Query layer | `.venv/bin/python scripts/build_duckdb.py` | `data/faffabout.duckdb` (views) plus the acceptance report | seconds |
| Optional | `.venv/bin/python scripts/fetch_pdb_metadata.py` | fills `outcomes.resolution` from RCSB | 2 min |

Flags:

| Script | Flag | Meaning |
|---|---|---|
| `01_fetch_zenodo.py` | `--force` | re-download even if the tarball verifies |
| | `--no-extract` | download and verify only |
| `02_parse_tt_xml.py` | `--single` | stream `tt.xml.gz` in one process instead of the 41 per-centre files (both hold exactly 335,771 targets) |
| | `--centres JCSG MCSG` | parse a subset |
| | `--workers N` | parallel workers (default 6) |
| | `--keep` | do not wipe `data/parquet/` first |
| `04_cluster_sequences.sh` | `--threads N` | MMseqs2 threads (default: all cores) |
| `build_duckdb.py` | `--check` | acceptance report only, no rebuild |

## 📊 Tables

All tables live in `data/parquet/` and are exposed as DuckDB views of the same name. Text columns that hold several values are `|`-joined.

| Table | Grain | Rows | Key columns |
|---|---|---|---|
| `targets` | one row per target | 335,771 | `target_id`, `centre`, `organism`, `taxon_id`, `sequence`, `seq_md5`, `seq_len`, `status_raw`, `stop_status`, `pdb_ids`, `first_seen`, `last_seen` |
| `target_sequences` | one row per target sequence (complexes carry several) | 385,139 | `target_id`, `seq_idx`, `is_primary`, `sequence`, `chem_type`, `construct_type` |
| `status_history` | one row per status event | 3,783,070 | `target_id`, `trial_id`, `history_id`, `status_raw`, `status_date`, `step_duration_days` |
| `status_history_canon` | `status_history` joined to the canonical map | 3,783,070 | adds `status_canon`, `stage_ord`, `method`, `terminal` |
| `trials` | one row per experimental attempt | 961,548 | `target_id`, `trial_id`, `status_raw`, `stop_status`, `protocol_refs`, `protocol_types`, `sequence`, `construct_type`, `free_text_notes` |
| `protocols` | one row per protocol (free text) | 1,501 | `protocol_id`, `centre`, `protocol_type`, `text` |
| `outcomes` | one row per PDB deposition | 11,954 | `target_id`, `pdb_id`, `method`, `resolution`, `deposit_date`, `source` |
| `target_stage` | one row per target | 335,771 | `max_stage`, `method`, `n_events`, `first_event`, `last_event`, `stopped` |
| `clusters` | one row per distinct protein sequence | 300,027 | `seq_md5`, `cluster_id`, `cluster_rep`, `cluster_size` |
| `target_summary` | view joining the above, one row per target | 335,771 | everything Phase 2 needs |

### Canonical ladder

`config/status_map.yaml` maps all 30 values of the TargetTrack status vocabulary (read from the archive's own `targetTrackEnumeratedDataItems-v1.4.1.xls` and cross-checked against the XSD) onto nine gates. NMR and EM milestones map onto gates 5 to 7 of their own ladders and `method` stays on the target, so the ladder is selectable at inference.

| `stage_ord` | `status_canon` | X-ray values | NMR values | EM values |
|---|---|---|---|---|
| 0 | selected | selected | | |
| 1 | cloned | cloned, expression tested | | |
| 2 | expressed | expressed | | |
| 3 | soluble | soluble, membrane protein solubilized | | |
| 4 | purified | purified, mass spec verified | | |
| 5 | crystallised | crystallized | HSQC satisfactory | EM images |
| 6 | diffracting | diffraction-quality crystals, diffraction, native diffraction-data, phasing diffraction-data | NMR assigned (backbone, sidechain) | EM reconstruction (in EMDB) |
| 7 | structure | crystal structure | NMR structure | EM fitted model |
| 8 | deposited | in PDB | in BMRB | |
| null | work_stopped, assay, test_target, other | never a stage; `work stopped` is the terminal marker | | |

"Expression tested" records that a test was run, not that protein was seen, so it does not advance past cloned. Every value seen in the data that is not in the map lands in `status_unmapped.csv` with its count rather than becoming a silent NULL.

## 🧪 Phase 1 acceptance

Run `scripts/build_duckdb.py`. Gate from the spec: `count(*)` from `targets` in the low 300,000s, `status_canon` has no NULLs outside the documented unmapped set, and the cluster file resolves every protein target.

Result on 2026-09-07:

```text
views built in data/faffabout.duckdb
targets                           335,771   (expect low 300,000s)
duplicate target_id                     0
status_history                  3,783,070
trials                            961,548
protocols                           1,501
outcomes                           11,954
target_sequences                  385,139
status_canon NULL rows                  0   over 0 distinct raw values (must all be in status_unmapped.csv)
documented unmapped values              0
protein targets with sequence     335,575
protein targets w/o cluster             0   (must be 0)
clusters (30% id)                  88,452

targets per centre (top 12):
 centre     n  mean_len  pct_deposited  pct_stopped
   MCSG 69579     260.0            2.3         32.5
   NESG 59953     347.0            1.8         57.9
   JCSG 40881     267.0            3.7          0.0
 NYSGRC 33193     404.0            0.7         43.0
   SGPP 20856     344.0            0.1          2.8
NYSGXRC 15927     432.0            5.6          0.4
  SECSG 15287     375.0            0.3          4.8
NYCOMPS 14930     378.0            0.1         76.3
 SSGCID 14517     369.0            4.9          0.0
    EFI 10567     366.0            1.9         41.9
  CSGID  9161     398.0            7.4         58.9
   CESG  8840     272.0            1.4         92.8

max_stage distribution:
max_stage      n   pct
        0 101189 30.14
        1  97013 28.89
        2  47737 14.22
        3  27163  8.09
        4  41655 12.41
        5   6067  1.81
        6   3950  1.18
        7    396  0.12
        8  10500  3.13
     none    101  0.03

PHASE 1 ACCEPTANCE: PASS
```

Every one of the 88,452 clusters is a unit for splitting: 50,114 are singletons and the largest holds 447 sequences. One thing the per-centre table already shows is that the `work stopped` marker is a centre convention rather than a scientific one (JCSG never uses it, CESG closes 93% of its targets with it), which is why Phase 2 derives censoring from activity dates and never from that flag.

Tests: `.venv/bin/python -m pytest -q` (status-map integrity and the parser on a synthetic target).

## 🎓 Roadmap and the science

The full plan is in `PROJECT_PLAN.md` and the specification in `faffabout_build_spec_v1.md`. The parts that matter most:

- **Censoring** (Phase 2). For each centre, the 99.5th percentile of its last activity date marks when it effectively stopped reporting. A target below gate 8 whose last activity falls within 180 days of that date, or of the 2017-06-30 freeze, is censored, not failed. Expect 15 to 20% of the archive.
- **Three label families.** L1 gate transitions (cleared, failed, censored) for every gate a target actually entered; L2 terminal outcome for calibration; L3 within-cluster preference pairs for an optional DPO stage.
- **Four negative-mining rules, in code.** Censored is never a negative. Not-attempted is never a negative. Hard negatives (same cluster, over 70% identity, divergent fate) are tagged and oversampled. Easy negatives are capped at 25% of the negative mass.
- **Three splits.** Cluster-held-out 80/10/10, leave-one-centre-out, and temporal (train before 2014, test from 2014).
- **GBM first.** The gradient-boosted baseline on the same features is expected to beat the LLM on AUC. The LLM earns its place with the narrative, attribution, construct recommendation and orthologue ranking, and any hallucinated target or PDB ID is an automatic eval failure.

## ✅ To Do

- [x] **Phase 1: acquire.** `01_fetch_zenodo.py` downloads, md5-verifies and extracts the 833 MB tarball, records the CC-BY-SA-4.0 licence and inventories `Documentation/` (schema XSD and PDF, controlled vocabulary spreadsheet, contributors list)
- [x] **Phase 1: parse.** `02_parse_tt_xml.py` streams all 41 per-centre XML files in parallel with `lxml.iterparse` (element names verified against the XSD, not guessed) into six Parquet tables plus a deduplicated FASTA; the per-centre files were verified to hold exactly the same 335,771 targets as `tt.xml.gz`
- [x] **Phase 1: normalise.** `03_normalise_status.py` reads the vocabulary spreadsheet, asserts every value is mapped in `config/status_map.yaml`, writes the canonical ladder and a per-target `max_stage` summary, and reports unmapped values explicitly
- [x] **Phase 1: cluster.** `04_cluster_sequences.sh` runs MMseqs2 at 30% identity and resolves every distinct protein sequence to a cluster id
- [x] **Phase 1: query layer and acceptance.** `build_duckdb.py` attaches DuckDB views over the Parquet directory and prints the acceptance report
- [x] **Phase 1: PDB resolution.** `fetch_pdb_metadata.py` fills `outcomes.resolution` from RCSB: 11,607 of 11,802 distinct ids resolve (the rest are obsolete), 9,440 carry a resolution
- [ ] **Phase 2: censoring.** Per-centre 99.5th-percentile last-activity date, 180-day window, global freeze; `censored` as a first-class boolean
- [ ] **Phase 2: labels.** L1 gate transitions, L2 terminal outcomes, L3 preference pairs, with the four negative-mining rules in code
- [ ] **Phase 2: features.** Composition, topology, annotation, construct and archive-context features; ESM-2 disorder on ZeroGPU cached by `seq_md5`; never the raw sequence in a prompt
- [ ] **Phase 2: splits and SFT.** Cluster, centre and temporal splits; five-task chat JSONL with 8 to 12 paraphrases per template, sorted by token length
- [ ] **Phase 3: GBM baseline.** LightGBM on the same features; the number the LLM must add over it
- [ ] **Phase 3: LoRA.** Llama-3.1-8B-Instruct 8-bit, all 32 layers, rank 16, `mask_prompt`, W&B; optional DPO on L3 pairs; fuse with `--de-quantize`
- [ ] **Phase 3: eval.** Brier, 10-bin ECE, per-gate AUROC, bottleneck top-1, GBM delta on all three splits; 40 graded narratives with zero hallucinated ids
- [ ] **Phase 4: serve.** Flask app with the Pipeline Rig front end, `POST /predict` contract, censoring visible at all times, live at faffabout.mdeller.com and listed on the mdeller.com launcher
- [ ] **Licence.** Choose the code licence (the data is CC-BY-SA-4.0; Llama 3.1 has its own community licence to check before any model redistribution)

---

## 👤 Author

**Marc C. Deller, D.Phil.**  
Structural biologist & drug discovery scientist  

<table>
<tr>
<td>🌐</td><td><a href="https://marcdeller.com" target="_blank" rel="noopener noreferrer">marcdeller.com</a></td>
<td>✉️</td><td><a href="mailto:marc@marcdeller.com">marc@marcdeller.com</a></td>
<td>🐙</td><td><a href="https://github.com/bellcheddar/FAFFAbout" target="_blank" rel="noopener noreferrer">github.com/bellcheddar/FAFFAbout</a></td>
</tr>
</table>
