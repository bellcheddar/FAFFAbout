# 🪦 FAFFAbout

> **Fine-tuned Attrition Forecasting From Archives: know where your protein is likely to die before you order the gene.**

![python](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white) ![lxml](https://img.shields.io/badge/lxml-6.1-467FF7) ![pyarrow](https://img.shields.io/badge/pyarrow-25.0-467FF7) ![duckdb](https://img.shields.io/badge/duckdb-1.5-FFF000?logo=duckdb&logoColor=black) ![pandas](https://img.shields.io/badge/pandas-3.0-150458?logo=pandas&logoColor=white) ![mmseqs2](https://img.shields.io/badge/MMseqs2-18-00897B) ![targets](https://img.shields.io/badge/targets-335%2C771-467FF7) ![status events](https://img.shields.io/badge/status%20events-3.78M-467FF7) ![clusters](https://img.shields.io/badge/clusters%20(30%25%20id)-88%2C452-467FF7) ![tests](https://img.shields.io/badge/pytest-116%20passing-00897B) ![data](https://img.shields.io/badge/data-PSI%20TargetTrack%20%C2%B7%20CC--BY--SA--4.0-9b51e0) ![phase 1](https://img.shields.io/badge/phase%201-complete-fcb900) ![censored](https://img.shields.io/badge/censored-19.03%25-9b51e0) ![phase 2](https://img.shields.io/badge/phase%202-complete-fcb900) ![lightgbm](https://img.shields.io/badge/LightGBM-4.7-00897B) [![MLX-LM](https://img.shields.io/badge/MLX--LM-Apple%20Silicon-000000?logo=apple&logoColor=white)](https://github.com/ml-explore/mlx-lm) ![author](https://img.shields.io/badge/author-Marc%20C.%20Deller%2C%20D.Phil.-1C244B)

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
| 6. Censoring | `.venv/bin/python scripts/censoring.py` | `data/parquet/censoring/` plus the censoring report | seconds |
| 7. Labels | `.venv/bin/python scripts/06_build_labels.py` | `data/parquet/labels_{l1,l2,l3}/` plus the label report | seconds |
| 8. Splits | `.venv/bin/python scripts/splits.py` | `data/parquet/splits/` plus the leakage report | seconds |
| 9. Features | `.venv/bin/python scripts/05_derive_features.py` | `data/parquet/features/` | 6 min |
| 10. SFT corpus | `.venv/bin/python scripts/07_build_sft.py` | `data/sft/{train,valid,test}.jsonl` | 2 min |

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
| `trials` | one row per experimental attempt | 961,548 | `target_id`, `trial_id`, `status_raw`, `stop_status`, `protocol_refs`, `protocol_types`, `sequence` (protein), `sequence_dna`, `construct_type`, `free_text_notes` |
| `labels_l1` | one row per (target, gate entered) | 906,573 | `gate`, `gate_name`, `label`, `in_loss`, `censored`, `max_stage` |
| `labels_l2` | one row per uncensored target | 271,619 | `outcome`, `deposited`, `max_stage` |
| `labels_l3` | one row per preference pair | 172,695 | `chosen_id`, `rejected_id`, `stage_gap`, `hard`, `cross_centre` |
| `censoring` | one row per target | 335,771 | `censored`, `censored_reason`, `censored_centre_winddown`, `censored_bulk_closure`, `last_any`, `wind_down`, `on_bulk_closure_date` |
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

## 🪦 Censoring: the result that decides the project

The specification's rule (per-centre 99.5th-percentile last-activity date, 180-day window) censors **5.22%** of the archive, against its own stated expectation of 15 to 20%. It misses the mass closures entirely, and those are the whole phenomenon: NESG stopped **34,605 targets on 2010-06-30**, 99.6% of every stop it ever recorded, while its 99.5th-percentile last-activity date sits at 2015-02-25, five years later. Not one of those targets falls inside a 180-day window.

FAFFAbout therefore applies a second rule alongside it. A **bulk closure** is a single date on which one centre stopped at least 200 targets amounting to at least 5% of all the stops it ever recorded. That is an administrative act, which is precisely the mechanism the specification describes, observed directly rather than inferred from a quantile.

| Rule | Targets | Share |
|---|---|---|
| Bulk closure only | 46,367 | 13.81% |
| Centre wind-down only | 14,627 | 4.36% |
| Both | 2,898 | 0.86% |
| **Censored (either)** | **63,892** | **19.03%** |
| Not censored | 271,879 | 80.97% |

Thirteen bulk-closure dates are detected across seven centres. The rule is validated by a column that was not used to build it: on a bulk-closure date, **93.4% of stop reasons are "other" or "duplicate target found"** and only 6.5% name an experimental failure, whereas every other stop in the archive names a specific scientific cause 68% of the time (expression failed 33.3%, cloning failed 14.3%, purification failed 12.7%). Administrative closure and experimental failure look completely different, and the rule separates them.

The per-centre picture is the reason none of this can be skipped. JCSG censors at 0% (it never used the `work stopped` marker and stayed active until 2016), NESG at 58.5%, CESG at 47.6%. A model trained without this correction would learn the funding history of US structural genomics, and it would learn it very well.

## 🎯 The label sets

`scripts/06_build_labels.py`. A gate is the transition **out of** a stage, so the nine-rung ladder has eight of them and stage 8 is terminal.

| Gate | Transition | Rows | Cleared | Failed | Censored | Cleared, of those in the loss |
|---|---|---|---|---|---|---|
| 0 | selected → cloned | 335,670 | 234,481 | 71,004 | 30,185 | 76.8% |
| 1 | cloned → expressed | 234,481 | 137,468 | 83,200 | 13,813 | 62.3% |
| 2 | expressed → soluble | 137,468 | 89,731 | 41,443 | 6,294 | 68.4% |
| 3 | soluble → purified | 89,731 | 62,568 | 20,083 | 7,080 | 75.7% |
| 4 | purified → crystallised | 62,568 | 20,913 | 36,805 | 4,850 | **36.2%** |
| 5 | crystallised → diffracting | 20,913 | 14,846 | 4,611 | 1,456 | 76.3% |
| 6 | diffracting → structure | 14,846 | 10,896 | 3,814 | 136 | 74.1% |
| 7 | structure → deposited | 10,896 | 10,500 | 318 | 78 | 97.1% |

906,573 rows, of which 842,681 enter the loss and 63,892 are held out as censored: exactly one per censored target, since a censored target still passed every gate below its last. The gradient is the one a crystallographer would predict. Getting a purified protein to crystallise is the wall at 36.2%, and once a structure exists depositing it is near-automatic at 97.1%.

**L2** is one row per uncensored target, `deposited` or `stalled_at_{g}`, for Brier score and expected calibration error: 271,619 rows at a 3.86% positive rate, against the specification's expected 4%.

**L3** is 172,695 within-cluster preference pairs, uncensored, differing by more than one gate. 72,505 are **hard** (the two proteins also share a 70%-identity cluster) and 37,760 of those are same-centre, which is the most valuable stratum in the corpus: the same lab, a near-identical protein, a different fate, so the difference is the construct or the host rather than the pipeline. The per-cluster cap keeps hard pairs first rather than an arbitrary slice.

## 🧬 Features

`scripts/05_derive_features.py`, with the sequence functions isolated in `features_seq.py` and cross-checked against Biopython (GRAVY matches exactly, isoelectric point within 0.6 units on differing pKa sets).

| Group | Features | Coverage |
|---|---|---|
| Composition | length, isoelectric point, GRAVY, net charge at pH 7, cysteine and methionine counts, aromatic, glycine and proline fractions | 99.9% |
| Topology | transmembrane helix count, signal peptide, FoldIndex disorder overall and by terminus, low-complexity fraction | 99.9% |
| Taxonomy | superkingdom, kingdom, phylum, genus from the NCBI dump | 98.3% |
| Construct | host, tag, protease, selenomethionine, autoinduction, codon optimisation, refolding, detergent | host 52.1%, tag 27.2% |
| Trial record | construct boundaries, expression level, solubility level, final concentration | boundaries 11.9%, expression 10.7% |
| Archive context | precedent counts, closest identity band, cluster base rate per gate, cluster censored fraction | 73.5% have a precedent |

Two things worth stating. The **archive-context features are computed over the training split only, with the target's own contribution subtracted back out**, because without that leave-one-out a target reads its own fate off its cluster's base rate and every metric flatters itself. And the **transmembrane predictor validates against the archive's own annotation**: targets the archive calls membrane proteins average 9.57 predicted helices against 0.78 for the rest, and 89.4% of them are called polytopic against 7.5%.

Host and tag come from mining the 1,501 shared protocol documents that 95.5% of trials point at, which is why a few hundred kilobytes of free text annotates half the archive.

## 📉 The baseline, and the leak it caught

`baseline/gbm_baseline.py`. The specification calls this non-negotiable and it earned that immediately: the first run scored a **mean AUROC of 0.985 across the gates and exactly 1.0000 on the terminal outcome**, on a clean cluster-held-out split. That is the meaningless result the specification warns about, and it arrived through the feature columns rather than through the split.

Three columns were the answer in disguise. A PDB reference is present for 49.5% of targets that cleared crystallisation and 0.1% of those that did not, because a deposition reference means it was deposited. The `method` field is derived from the status history, so `xray` carries a mean maximum stage of 6.68 against 1.40 for `none`. Successful targets average 16.4 trials against 4.1. `config/features.yaml` now separates what a scientist knows **before ordering the gene** from what was recorded **while the attempt ran**, and a test reads the boundary back off the saved boosters.

A subtler one: host, tag and protease are mined from the protocol a trial referenced, so their **presence** tracks progression. At the first gate, host is missing for 96.2% of failures and 27.3% of successes, because a target that died at selection never had an expression protocol attached. They are excluded from the headline and evaluated separately on rows that all declare a protocol, where only the choice varies.

**Cluster-held-out, 37 prediction-time features:**

| Gate | Transition | n | Base rate | AUROC | Brier | ECE |
|---|---|---|---|---|---|---|
| 0 | selected → cloned | 30,890 | 0.763 | 0.883 | 0.110 | 0.028 |
| 1 | cloned → expressed | 22,062 | 0.570 | 0.770 | 0.194 | 0.040 |
| 2 | expressed → soluble | 12,112 | 0.639 | 0.869 | 0.141 | 0.023 |
| 3 | soluble → purified | 7,130 | 0.751 | 0.841 | 0.138 | 0.030 |
| 4 | purified → crystallised | 5,005 | 0.344 | 0.762 | 0.179 | 0.015 |
| 5 | crystallised → diffracting | 1,639 | 0.798 | 0.826 | 0.126 | 0.025 |
| 6 | diffracting → structure | 1,294 | 0.701 | 0.853 | 0.139 | 0.041 |
| 7 | structure → deposited | 901 | 0.981 | 0.906 | 0.016 | 0.012 |
| | terminal (deposited) | 30,890 | 0.029 | 0.866 | 0.024 | 0.007 |

**All four configurations:**

| Configuration | Features | Mean AUROC | Terminal AUROC |
|---|---|---|---|
| Cluster-held-out (headline) | 37 | 0.839 | 0.866 |
| Temporal, train pre-2014 | 37 | 0.740 | 0.849 |
| Declared protocol, restricted | 42 | 0.851 | 0.903 |
| Post-hoc added back (negative control) | 59 | 0.985 | 1.000 |

The last row is kept as a **negative control** and asserted by a test, because a check that has only ever passed is not a check. The gap between the first row and the last is what the post-hoc block gives away.

Archive-context features deserve one caveat. Under the cluster split they are structurally absent at test time: whole clusters are held out, so 0.0% of test targets have a precedent against 91.9% of training targets. The temporal split is the one where precedent is both available (82.3% of test targets) and honest, since a 2014 target may have pre-2014 precedents in its own cluster.

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
- [x] **Phase 2: censoring.** Two rules in `scripts/censoring.py`, both stored separately and configurable in `config/labels.yaml`: the specification's per-centre wind-down (99.5th-percentile last activity, 180-day window, plus the global freeze) and a new bulk-closure detector. 19.03% of the archive is censored, inside the 12 to 22% gate
- [x] **Phase 2: labels.** `scripts/06_build_labels.py` builds L1 (906,573 gate transitions over eight gates, 842,681 in loss), L2 (271,619 uncensored terminal outcomes, 3.86% deposited) and L3 (172,695 within-cluster preference pairs, 72,505 of them hard at >70% identity)
- [x] **Phase 2: features.** `scripts/05_derive_features.py` and `features_seq.py`: composition (pI, GRAVY, charge, cysteines, aromaticity, low complexity), topology (transmembrane helices, signal peptide, FoldIndex disorder by terminus), taxonomy for 98.9% of targets, host and tag mined from the 1,501 shared protocol documents, and leave-one-out archive context. The raw sequence never enters a prompt
- [ ] **Phase 2: ESM-2 features.** Disorder and embeddings on ZeroGPU, cached by `seq_md5`, to replace the local FoldIndex proxy
- [x] **Phase 2: splits.** `scripts/splits.py`: cluster-held-out at exactly 80/10/10 with zero cluster and zero sequence leakage, leave-one-centre-out across five centres, and a temporal split that excludes the 7,846 targets straddling the 2014 boundary
- [x] **Phase 2: SFT corpus.** `scripts/07_build_sft.py`: 120,000 training records in the specification's five-task mix, 2,000 each for validation and test drawn from held-out clusters, 10 paraphrases per template, sorted by token length for MLX padding
- [x] **Phase 3: GBM baseline.** `baseline/gbm_baseline.py`: LightGBM per gate and on the terminal outcome, across the cluster, temporal and declared-protocol configurations, with a leak demonstration as a negative control. Mean AUROC 0.839 cluster-held-out, terminal 0.866
- [x] **Phase 3: feature provenance.** `config/features.yaml` separates prediction-time from post-hoc features, after the first baseline scored a meaningless 0.985
- [ ] **Phase 3: disk and licence.** The model pipeline needs roughly 40 GB and 17 GB is free; Llama 3.1 is a gated repository needing the Meta community licence accepted
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
