# 🪦 FAFFAbout

> **Fine-tuned Attrition Forecasting From Archives: know where your protein is likely to die before you order the gene.**

![python](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white) ![duckdb](https://img.shields.io/badge/duckdb-1.5-FFF000?logo=duckdb&logoColor=black) ![lightgbm](https://img.shields.io/badge/LightGBM-4.7-00897B) ![mlx-lm](https://img.shields.io/badge/mlx--lm-0.31-000000?logo=apple&logoColor=white) ![mmseqs2](https://img.shields.io/badge/MMseqs2-18-00897B) ![flask](https://img.shields.io/badge/flask-3.1-000000?logo=flask&logoColor=white) ![targets](https://img.shields.io/badge/targets-335%2C771-467FF7) ![censored](https://img.shields.io/badge/censored-19.03%25-9b51e0) ![tests](https://img.shields.io/badge/pytest-202%20passing-00897B) ![data](https://img.shields.io/badge/data-PSI%20TargetTrack%20%C2%B7%20CC--BY--SA--4.0-9b51e0) ![status](https://img.shields.io/badge/status-GBM%20%2B%20archive%20evidence-00897B) ![author](https://img.shields.io/badge/author-Marc%20C.%20Deller%2C%20D.Phil.-1C244B)

<table>
<tr>
<td>🌐 <b>Website</b></td><td><a href="https://marcdeller.com" target="_blank" rel="noopener noreferrer">marcdeller.com</a></td>
<td>✉️ <b>Contact</b></td><td><a href="mailto:marc@marcdeller.com">marc@marcdeller.com</a></td>
<td>🐙 <b>GitHub</b></td><td><a href="https://github.com/bellcheddar/FAFFAbout" target="_blank" rel="noopener noreferrer">bellcheddar/FAFFAbout</a></td>
</tr>
</table>

---

![The FAFFAbout Pipeline Rig showing a forecast for UniProt P0A6Y8, the E. coli chaperone DnaK. Eight gate lamps run from selected through to deposited; the first five are green and crystallised is lit amber as the predicted wall, with cumulative survival falling from 1.00 to 0.18 there. The console on the left carries the sequence, method, centre, host, tag and protease. Below the lamps, a panel headed What close relatives tried states that 1 of the 4 close relatives that reached purified got through to crystallised and that full-length constructs got further than truncated ones, then lists the most informative relatives with their constructs and how far each trial got. A precedent table with censored records greyed and dashed, counterfactual buttons and the caveats follow.](docs/screenshots/rig.png)

## 💡 In one minute

Getting from a gene to a protein structure is a long chain of experiments: clone the gene, express the protein, get it soluble, purify it, crystallise it, collect diffraction data, solve the structure, deposit it. Most attempts fail somewhere along the way, and the failures are expensive in time and money.

FAFFAbout forecasts **where along that chain a given protein is most likely to fail**, before any lab work starts. You paste a protein (a FASTA sequence, a UniProt accession or a PDB ID) and it returns:

- the chance of getting through each of the eight steps, and which step is the likeliest wall;
- the real archive proteins most similar to yours, and how far each of them got;
- what changing the expression host or tag would do to the odds;
- what close relatives in the archive actually tried (constructs, hosts, tags, how far each attempt got and why it stopped), and an honest statement of how much evidence there is.

The forecasts are learned from the **Protein Structure Initiative's TargetTrack archive**: 335,771 proteins attempted by 41 structural genomics centres between 2000 and 2017, with 961,548 experimental trials and 3.8 million recorded status changes. It is the only large record of where protein production attempts *stopped*, not just which ones succeeded.

**Useful for:** triaging a target list before ordering genes, choosing between orthologues, picking a host and tag on evidence rather than habit, and setting realistic expectations with collaborators.

## 🧭 What it is not

It is not a structure predictor and not a replacement for AlphaFold. It predicts *experimental attrition*: whether a real lab attempt gets through each step, which is a different quantity. The archive also has a selection bias: PSI targets were mostly chosen as likely-soluble bacterial proteins, so forecasts for membrane and eukaryotic proteins are extrapolation, and the interface says so.

## ⚙️ How it works

Every forecast has two parts, and neither is generated text.

1. **The numbers come from a GBM** (explained below): the probability of getting through each step, the likeliest wall, and what changing the host or tag would do.
2. **The evidence comes straight from the archive.** FAFFAbout finds the proteins most similar to yours (MMseqs2, at least 30% identity) and shows what was actually tried on them: which constructs and residue ranges, which hosts and tags, the expression and solubility each trial recorded, how far each got, and the reason its centre recorded when it stopped. It leads with plain facts about the likeliest wall, such as "1 of the 4 close relatives that reached purified got through to crystallised" or "full-length constructs got further than truncated ones", and shows first the relatives that met the same wall.

### What is a GBM?

**GBM stands for gradient-boosted model.** It is a standard, well-understood machine-learning technique for data that comes in rows and columns. It builds hundreds of small decision trees, each one learning to correct the mistakes of the trees before it, and adds their answers together into a single prediction.

In FAFFAbout there is one GBM for each step of the pipeline. Each takes a protein's features (its length, cysteine count, predicted disorder, source organism, and how its relatives in the archive fared) and returns the probability that the protein gets through that step. Chaining the eight steps together gives the forecast: the probability of surviving to each stage, and the **bottleneck**, the step where the most probability is lost.

**Why the numbers come from the GBM.** A fine-tuned language model was also tried (see below). Measured on proteins neither had seen:

| | GBM | Language model (round 04) |
|---|---|---|
| Ranking successes above failures (AUROC; 0.5 is chance, 1.0 is perfect) | **0.857** | 0.638 |
| Calibration error (lower is better) | **0.024** | 0.036 |
| Names the step where the protein really stopped | **34.6%** | 27.2% |

So the GBM supplies every number on the page.

## 🪦 The trap in the data: censoring

A protein recorded as stopping at "cloned" in 2017 could mean three different things. It failed on scientific grounds; its centre's funding ended and the file was simply closed; or it was still in progress when the archive froze. A model that cannot tell these apart learns to predict **when US funding programmes ended**, not which proteins are hard.

FAFFAbout identifies these **censored** records from the data itself and never counts them as failures. Two rules find them:

| Rule | What it catches | Targets |
|---|---|---|
| Centre wind-down | Activity within 180 days of a centre's effective last date, or of the 2017 freeze | 17,525 |
| Bulk closure | A single date on which a centre closed at least 200 targets and at least 5% of everything it ever stopped | 49,265 |
| **Either (censored)** | | **63,892 (19.03%)** |

The bulk-closure rule is needed because the wind-down rule alone catches only 5.2% of the archive. NESG, for example, closed 34,605 targets on a single day (30 June 2010), five years before its last activity. The rule is validated on a column it never used: on bulk-closure dates, 93.4% of recorded stop reasons are "other" or "duplicate target found", against 68% naming a specific experimental failure on every other stop.

Censored records are kept everywhere a scientist would want to see them (in the precedent table, drawn grey and dashed) and excluded everywhere they would mislead (every training loss).

## 📈 How good is it?

**Per-step accuracy of the GBM**, on protein families held out from training entirely:

| Step | Base rate of success | AUROC |
|---|---|---|
| Selected → cloned | 76% | 0.883 |
| Cloned → expressed | 57% | 0.770 |
| Expressed → soluble | 64% | 0.869 |
| Soluble → purified | 75% | 0.841 |
| Purified → crystallised | **34%** | 0.762 |
| Crystallised → diffracting | 80% | 0.826 |
| Diffracting → structure | 70% | 0.853 |
| Structure → deposited | 98% | 0.906 |
| Overall: reaches the PDB | 3% | 0.866 |

The pattern is the one a crystallographer would predict: crystallisation is the wall, and once a structure exists, deposition is near-automatic.

**Naming the bottleneck.** On 254 proteins from 2014 onwards whose true stopping point is known (models trained only on earlier data):

| Method | Right |
|---|---|
| Always guess the most common wall | 22.8% |
| The GBM, lowest single-step probability | 18.5% |
| **The GBM, largest loss of survival (what the app shows)** | **34.6%** |

**The language-model experiment, and why it was dropped.** The original plan had a fine-tuned language model (Llama 3.1 8B) write an explanation beside each forecast. Five rounds of training got it to the point where it was safe and faithful by every automated check:

| Criterion | Round 04 | Round 5 |
|---|---|---|
| Invents a target or PDB identifier | 0 of 40 | **0 of 40** |
| Names the same bottleneck as the GBM | no (27.2% vs 34.6% right) | **yes, on all 235 test proteins** |
| Cites a precedent from the archive | 8 of 40 | **36 of 40** |
| Flags censored precedents when present | 19 of 20 | **29 of 30** |

But an expert asked of 10 fresh forecasts whether the explanation would change what they did judged it useful in **0 of 10**. The model had been trained on template-written answers, so the best it could do was restate the forecast in stock sentences. It was replaced by the archive panel above, which shows the evidence itself instead of prose about it. The training code and every evaluation remain in the repository and in `PROJECT_PLAN.md`.

## 🖥️ The application

One field takes a FASTA sequence, a UniProt accession or a PDB ID with an optional chain. The resolved sequence is always shown before anything is computed, so a mis-resolved identifier cannot silently produce a confident forecast for the wrong protein.

A request searches all 300,027 distinct archive sequences with MMseqs2 (about 0.8 s), joins the hits to their outcomes and censoring flags, computes the sequence features, scores each step with its GBM, and reads the trial records of the retrieved relatives for the evidence panel. **Nothing in the browser computes a probability, and nothing on the page is generated text.** The **counterfactual** buttons re-run the forecast with one choice changed (host, tag, protease, codon optimisation, auto-induction), using a second GBM trained only on records that declare a protocol.

| Endpoint | Does |
|---|---|
| `GET /` | the Pipeline Rig; `?q=P0A6Y8&host=arctic&tag=his` renders a shareable forecast |
| `POST /api/resolve` | whatever was pasted, resolved to a sequence and shown before computing |
| `POST /api/predict` | the full forecast |
| `GET /api/archive` | per-centre outcome counts for the archive map |
| `GET /healthz` | each component's status, reported separately |

![The archive map view: one column per contributing centre across the whole PSI TargetTrack archive, ordered by size from MCSG and NESG down to the smallest centres. Each column is split by fate, with deposited in green at the base, censored in grey and stalled in red. A panel reports 335,771 targets, 10,500 deposited and 19.0% censored.](docs/screenshots/archive_map.png)

## 🔧 Installation and building from scratch

```bash
cd FAFFAbout
uv venv --python 3.14 .venv        # or: python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
brew install mmseqs2
```

Everything under `data/`, `adapters/` and `models/` is regenerated from the public archive (Zenodo [10.5281/zenodo.821654](https://doi.org/10.5281/zenodo.821654), CC-BY-SA-4.0) and is never committed. Run the steps in order:

| Step | Command | Produces |
|---|---|---|
| Fetch | `scripts/01_fetch_zenodo.py` | the 795 MiB archive, md5-verified, licence recorded |
| Parse | `scripts/02_parse_tt_xml.py --workers 8` | Parquet tables for targets, trials, status history, protocols, outcomes |
| Normalise | `scripts/03_normalise_status.py` | every status mapped onto the nine-stage ladder |
| Cluster | `scripts/04_cluster_sequences.sh` | MMseqs2 families at 30% identity |
| Query layer | `scripts/build_duckdb.py` | `data/faffabout.duckdb` and an acceptance report |
| Censoring | `scripts/censoring.py` | the censoring flags and their report |
| Labels | `scripts/06_build_labels.py` | per-step outcomes, terminal outcomes, preference pairs |
| Splits | `scripts/splits.py` | family-held-out, leave-one-centre-out and temporal splits |
| Features | `scripts/05_derive_features.py` | sequence, taxonomy, construct and archive-context features |
| GBM | `baseline/gbm_baseline.py` | the per-step boosters the app uses |

The language-model experiment is reproducible but not needed by the app: `scripts/06b_gbm_forecasts.py` (out-of-fold GBM forecasts for training prompts), `scripts/07_build_sft.py` (the corpus), `scripts/08_train.py` (LoRA training; run `scripts/preflight.sh` first) and `scripts/evaluate_round.sh` (the four automatic checks).

Prefix each Python step with `.venv/bin/python`. Tests: `.venv/bin/python -m pytest -q`.

## 🔬 For the technically curious

**The ladder.** All 30 TargetTrack status values map onto nine stages (`config/status_map.yaml`), read from the archive's own controlled vocabulary. A *gate* is the transition out of a stage, so there are eight. NMR and cryo-EM milestones map onto their own ladders. "Expression tested" records that a test ran, not that protein was seen, so it does not advance past cloned. `work stopped` is never a stage or a failure label: its use is a centre convention (JCSG never uses it; CESG closes 93% of its targets with it).

**Labels.** L1 is one row per gate a target actually entered (906,573 rows; 842,681 used in training, the rest censored). L2 is the terminal outcome per uncensored target (271,619, 3.86% deposited). L3 is 172,695 within-family preference pairs, 72,505 of them near-identical proteins (over 70% identity) with different fates.

**Features.** Composition (length, isoelectric point, GRAVY, charge, cysteines), topology (transmembrane helices, signal peptide, disorder by terminus, low complexity), taxonomy, construct choices mined from 1,501 shared protocol documents, and archive context (how the protein's relatives fared), computed over the training split only with each target's own contribution removed. The raw sequence never enters a language-model prompt.

**Splits.** Always by MMseqs2 family (30% identity), never by row, because the same protein appears across many orthologues. The temporal split (train before 2014, test from 2014) is the honest test of forecasting the future and the one the narrative evaluations grade.

**The leak the baseline caught.** The first GBM scored a near-perfect AUROC of 0.985. Three columns were the answer in disguise: a PDB reference exists only for deposited targets, the `method` field is derived from the status history, and successful targets simply have more trials. `config/features.yaml` now separates what a scientist knows *before ordering the gene* from what was recorded *while the attempt ran*, and the leaky configuration is kept as a negative control asserted by a test.

**The evidence panel.** `app/relatives.py` reads every trial of every retrieved relative from DuckDB: the furthest stage each trial reached (from the canonical status history), the construct type and residue range, expression and solubility levels and final concentration where the centre recorded them, and the stop reason and remark. The facts at the top are computed over all retrieved relatives and all their trials, never just the ones displayed; censored relatives are excluded from every count and reported separately; administrative stop reasons ("other", "duplicate target found") are never given as the cause. Coverage varies by centre (MCSG records no construct types, for example), so the panel states only what was recorded.

**The language-model experiment.** Llama 3.1 8B Instruct (8-bit), LoRA rank 16 on all 32 layers, learning rate 2e-5, trained on Apple silicon with MLX. Round 5 trained it to narrate the GBM's out-of-fold forecast rather than invent one. Its evaluation (`scripts/evaluate_round.sh`) verified by output that the adapter was attached, then checked calibration, invented identifiers, whether the prose tracked its input, and bottleneck accuracy against where proteins really stopped. It passed all of those and failed the one that mattered: whether a scientist would act on it.

## 🗺️ Status and next steps

- [x] **Data:** archive parsed, normalised, clustered and queryable; censoring derived (19.03%)
- [x] **Labels, features and splits:** leakage-free by construction and checked by tests
- [x] **GBM:** mean AUROC 0.839 on held-out families; ships in the app as the source of every number
- [x] **Application:** built and working locally
- [x] **Evidence panel:** what close relatives actually tried, read from the archive's trial records
- [x] **Language-model experiment:** five rounds; safe and faithful but judged useful in 0 of 10 cases, so dropped from the app
- [ ] **ESM-2 features** to replace the local disorder predictor
- [ ] **Deploy** to faffabout.mdeller.com
- [ ] **Licence** for the code (the data is CC-BY-SA-4.0)

The full plan and a dated log of every decision are in `PROJECT_PLAN.md`; the original specification is `faffabout_build_spec_v1.md`.

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
