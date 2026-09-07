# FAFFAbout: Project Plan

> **Fine-tuned Attrition Forecasting From Archives.** Enter a FASTA, UniProt accession or PDB ID and get a breakdown of the protein plus calibrated guidance on where comparable targets historically died, grounded in roughly 330,000 real attempts from the PSI TargetTrack archive.

Source of truth for scope: `faffabout_build_spec_v1.md`. This file tracks the four build phases, what each has to deliver, the gate it must pass, and the live status. Update the status column in the same commit that changes it.

## Status

| Phase | Deliverable | Budget | Gate | Status |
|---|---|---|---|---|
| 1 | DuckDB + Parquet core, MMseqs2 clusters | 1 day | 330k targets queryable, statuses canonicalised, every target resolves to a cluster | complete 2026-09-07 (335,771 targets, 30/30 statuses mapped, 88,452 clusters, acceptance PASS) |
| 2 | L1/L2/L3 label sets, features, three splits, SFT JSONL | 2 to 3 days | Censoring rate 12 to 22%, no cluster leakage, 50 records pass expert eyeball | complete 2026-09-07 apart from Marc's eyeball of 50 records |
| 3 | GBM baseline, LoRA adapter, calibration + generative eval | 2 days plus overnight runs | LLM narrative adds something the GBM cannot, zero hallucinated IDs | not started |
| 4 | Flask app on faffabout.mdeller.com | 2 days | End-to-end FASTA in, full breakdown out | not started |

## The one idea that has to survive every phase

The PSI archive records where attempts *stopped*, not just which ones succeeded. A target parked at `cloned` on 30 June 2017 is one of three things: failed on scientific grounds, censored because its centre's funding ended, or still in flight when the archive froze. Treating all three as negatives trains a model to predict when US funding programmes ended. Censoring is derived empirically per centre, stored as a first-class boolean, excluded from every loss, kept in every retrieval context, and shown in the UI at all times.

## Phase 1: Parse and normalise the archive

**Goal:** turn one 795 MB tarball into a queryable relational core.

| Step | Script | Output | Notes |
|---|---|---|---|
| Acquire | `scripts/01_fetch_zenodo.py` | `data/raw/TargetTrack/`, `data/raw/zenodo_record.json` | Zenodo 10.5281/zenodo.821654, md5-verified, licence CC-BY-SA-4.0 recorded |
| Parse | `scripts/02_parse_tt_xml.py` | `data/parquet/{targets,status_history,trials,outcomes}/*.parquet`, `data/parquet/targets.fasta` | `lxml.iterparse` streaming with element clearing. Never DOM-parse `tt.xml.gz`. Tag names verified against `targetTrack-v1.4.1.pdf`, not guessed |
| Normalise | `scripts/03_normalise_status.py` | `data/parquet/status_map.parquet`, `data/parquet/status_unmapped.csv` | Reads the Documentation enumerations spreadsheet. Canonical `stage_ord` 0 to 8. Explicit unmapped report, never silent NULLs |
| Cluster | `scripts/04_cluster_sequences.sh` | `data/clusters/tt30_clusters.parquet` | MMseqs2 easy-cluster at 30% identity, 80% coverage of the shorter sequence |
| Query layer | `scripts/build_duckdb.py` | `data/faffabout.duckdb` | Views over the Parquet directory, read-only at runtime |

**Canonical ladder (`stage_ord`):** 0 selected, 1 cloned, 2 expressed, 3 soluble, 4 purified, 5 crystallised, 6 diffracting, 7 structure, 8 deposited. NMR and cryo-EM targets map gates 5 to 7 onto their own ladders; `method` stays on the target so the ladder is selectable at inference.

**Acceptance test:** `SELECT count(*) FROM targets` is in the low 300,000s; `status_canon` has no NULLs outside the documented unmapped set; the cluster file resolves every `target_id`.

## Phase 2: Labels, features, splits

**Goal:** the positive and negative label set. This phase decides whether the project works. Do not compress it.

1. **Censoring** (`scripts/06_build_labels.py`): per-centre 99.5th-percentile last activity date; a target is censored if `max_stage < 8` and its last activity falls within 180 days of that date, or within 180 days of the 2017-06-30 freeze. Expect 15 to 20% censored.
2. **Three label families:**
   - L1 gate transitions (45% of the SFT loss): one row per gate the target actually entered, labelled `cleared`, `failed` or `censored`. Not-attempted rows are never emitted.
   - L2 terminal outcome: `deposited` or `stalled_at_{g}` per uncensored target, for Brier and ECE.
   - L3 within-cluster preference pairs: uncensored pairs with `max_stage(A) > max_stage(B) + 1`, capped at 20 per cluster, for the optional DPO stage.
3. **Negative mining rules, in code:** censored is never a negative; not-attempted is never a negative; hard negatives (same cluster, >70% identity, divergent fate) tagged and oversampled 3x; easy negatives capped at 25% of negative mass, stratified on (TM count, kingdom, length decile).
4. **Features** (`scripts/05_derive_features.py`, `config/features.yaml`): composition, topology, annotation, construct, archive context. Never the raw sequence in the prompt. ESM-2 disorder and embeddings run on ZeroGPU and cache to Parquet keyed on `seq_md5`.
5. **Splits:** cluster-held-out 80/10/10; leave-one-centre-out across JCSG, NESG, MCSG, NYSGRC, CESG with `centre` as an input feature; temporal train pre-2014, test 2014 onward.
6. **SFT records** (`scripts/07_build_sft.py`): chat JSONL, five task types (gate judgement 45%, full forecast 20%, orthologue ranking 15%, construct recommendation 12%, failure attribution 8%), 8 to 12 paraphrases per template, sorted by token length.

**Acceptance test:** censored fraction 12 to 22%; hard-negative pairs in the tens of thousands; no cluster in more than one split; 50 random SFT records read as sensible to a domain expert.

## Phase 3: Baseline, then fine-tune

1. **GBM baseline first** (`baseline/gbm_baseline.py`): LightGBM on the same tabular features, L2 terminal and per-gate L1. Expect it to beat the LLM on AUC. It ships inside the app as the source of every probability number.
2. **LoRA** (`config/train_config.yaml`, `scripts/08_train.py`): Llama-3.1-8B-Instruct, 8-bit, all 32 layers, rank 16, `batch_size` 4, LR 1e-4 cosine, `mask_prompt: true`, W&B reporting. Budget 6 to 10 hours per run on M1. Run `scripts/preflight.sh` before every launch.
3. **Optional DPO** on L3 pairs via `mlx-lm-lora` if the SFT model is overconfident on sparse classes.
4. **Fuse** with `--de-quantize`; serve via `mlx_lm.server` on 127.0.0.1:8080 (GGUF export possible because the base is Llama).
5. **Eval** (`eval/eval_calibration.py`, `eval/eval_generative.py`): Brier, 10-bin ECE, AUROC per gate, bottleneck top-1, GBM delta, on all three splits. 40 held-out narratives graded against a rubric. Any hallucinated target or PDB ID is an automatic fail.

**Acceptance test:** a stated number the LLM adds over the GBM; zero hallucinated IDs.

## Phase 4: Serve

`nginx -> gunicorn -> Flask (app/server.py)` with `retrieve.py` (MMseqs2 search + DuckDB), `features.py`, `gbm.py`, `llm.py`. One input field accepts FASTA, UniProt accession or PDB ID with optional chain; resolved sequence always shown before computing. `POST /predict` returns the contract in spec section 7.3. The narrative is the only field the LLM writes. Front end is the Pipeline Rig (`faffabout_rig_v1.html` ported to `app/templates/rig.html`) with the Necropolis archive map as a secondary mode. Censored precedents render grey and dashed, never red.

**Acceptance test:** FASTA in, full breakdown out, on faffabout.mdeller.com.

## Known risks

| Risk | Mitigation |
|---|---|
| Model learns when US funding ended | Phase 2 censoring, the highest-priority correctness issue |
| Homology leakage inflates metrics | Cluster-level splits only, at 30% identity |
| Centre effects dominate | `centre` as an explicit feature plus leave-one-centre-out eval |
| Selection bias in the archive | PSI targets already passed a soluble-prokaryotic filter; membrane and eukaryotic predictions are extrapolation and the UI says so |
| Hallucinated precedent IDs | Never fine-tune retrievable facts; automatic eval fail |
| M1 training slower than expected | Drop to `num_layers: 16` and `batch_size: 2` before dropping to 4-bit |
| Local disk (26 GB free on 2026-09-07) | The 8-bit base alone is ~9 GB and the de-quantised fuse ~16 GB; clear space before Phase 3 |

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-09-07 | Repository folder is `FAFFAbout/` (spec says `faffabout/`) | APFS is case-insensitive; the folder already existed with the spec and rig in it |
| 2026-09-07 | Python 3.14 in `.venv` via `uv`, same as TopPDBLX and GOBSMACKED | House convention |
| 2026-09-07 | DuckDB over Parquet, not SQLite | Static 330k-row analytical store; group-bys an order of magnitude faster |
| 2026-09-07 | Parse the 41 per-centre XML files in parallel rather than streaming `tt.xml.gz` | Verified identical (335,771 targets both ways); 62 s on 8 workers instead of minutes |
| 2026-09-07 | `expression tested` maps to stage 1 (cloned), not 2 | It records that a test ran, not that protein was seen; revisit if Phase 2 shows it tracks `expressed` |
| 2026-09-07 | `work stopped` is a terminal marker, never a stage and never a failure label | Its use is a centre convention: JCSG 0%, CESG 93% of targets. Censoring comes from activity dates |
| 2026-09-07 | Censoring uses TWO rules: the spec's centre wind-down OR a new bulk-closure detector | The spec's rule alone censors 5.22%, against the spec's own stated expectation of 15 to 20% and its 12 to 22% gate. It misses mass closures outright: NESG stopped 34,605 targets on 2010-06-30 (99.6% of every stop it recorded) while its 99.5th-percentile last-activity date is 2015-02-25. Together the rules censor 19.03%. Validated independently: on bulk-closure dates 93.4% of stop reasons are "other" or "duplicate target found", against 68% naming a specific experimental failure on every other stop. Both flags are stored separately and either can be disabled in `config/labels.yaml` |
| 2026-09-07 | Trial construct sequences are chosen by chemistry, not position | Whole centres list the gene first: 90,746 trials (9.4%) had a nucleotide sequence in the protein column at ~3x the true length (NYCOMPS mean 1,121 vs 388). 16,931 carried no `sequenceChemicalType` element at all, so composition decides and the label is only a hint. The DNA is kept in its own column rather than discarded, since codon optimisation is a Phase 2 feature |
| 2026-09-07 | Out-of-range dates are nulled, never clamped | 5,244 events fall outside 1995 to the freeze (CSGID 5,136 on 1979-01-01, one in the year 0013, MPSBC quantile landing in 2041). Clamping would drag a centre's wind-down quantile backwards and mis-censor its whole cohort |
| 2026-09-07 | L1 emits eight gates (0 to 7), not nine | A gate is the transition out of a stage, so a nine-rung ladder has eight of them. Applying the spec's rule literally at gate 8 labelled all 10,500 deposited targets `failed` at the final gate, which would teach the model that deposition always fails |
| 2026-09-07 | L3 pairs are capped per cluster by hardest-first, not arbitrarily | The cap exists to stop large families dominating, but a random slice would discard the >70%-identity stratum the spec values at ~50 random negatives. Ordering by (hard, stage gap) keeps it: 72,505 of 172,695 pairs are hard |
| 2026-09-07 | SFT probabilities are the shrunk cluster base rate, never a function of the target's own outcome | Training 0.85 for every success and 0.15 for every failure teaches confident guessing, which is the overconfidence the spec predicts. The corpus's own targets measure at an expected calibration error of 0.019 |
| 2026-09-07 | Identifiers are never truncated in a precedent table | Printing `NYCOMPS-GO.78` in the prompt while the completion said `NYCOMPS-GO.7810` taught the model to extend an identifier it was given, which is the habit behind hallucinated precedent IDs. Caught by a test, not by reading |
| 2026-09-07 | The easy-negative cap applies to the FINAL negative mass, after hard oversampling | Rules 3 and 4 interact. Capping distinct rows would leave easy negatives at 10%; an earlier floor of "25% of all negatives" defeated the cap and let them reach 44% |
| 2026-09-07 | Kingdom comes from matching lineage NAMES, not the rank label | NCBI renamed `superkingdom` to `domain` and inserted new kingdom-level clades, so reading the rank returned "Metazoa" where "Eukaryota" was meant. Name matching resolved 98.9% of targets, 54.7% of them by organism name where no taxon id existed |
