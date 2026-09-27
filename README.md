# 🪦 FAFFAbout

> **Fine-tuned Attrition Forecasting From Archives: know where your protein is likely to die before you order the gene.**

![python](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white) ![duckdb](https://img.shields.io/badge/duckdb-1.5-FFF000?logo=duckdb&logoColor=black) ![lightgbm](https://img.shields.io/badge/LightGBM-4.7-00897B) ![mlx-lm](https://img.shields.io/badge/mlx--lm-0.31-000000?logo=apple&logoColor=white) ![mmseqs2](https://img.shields.io/badge/MMseqs2-18-00897B) ![flask](https://img.shields.io/badge/flask-3.1-000000?logo=flask&logoColor=white) ![targets](https://img.shields.io/badge/targets-335%2C771-467FF7) ![censored](https://img.shields.io/badge/censored-19.03%25-9b51e0) ![tests](https://img.shields.io/badge/pytest-194%20passing-00897B) ![data](https://img.shields.io/badge/data-PSI%20TargetTrack%20%C2%B7%20CC--BY--SA--4.0-9b51e0) ![status](https://img.shields.io/badge/status-round%205%20evaluated-00897B) ![author](https://img.shields.io/badge/author-Marc%20C.%20Deller%2C%20D.Phil.-1C244B)

<table>
<tr>
<td>🌐 <b>Website</b></td><td><a href="https://marcdeller.com" target="_blank" rel="noopener noreferrer">marcdeller.com</a></td>
<td>✉️ <b>Contact</b></td><td><a href="mailto:marc@marcdeller.com">marc@marcdeller.com</a></td>
<td>🐙 <b>GitHub</b></td><td><a href="https://github.com/bellcheddar/FAFFAbout" target="_blank" rel="noopener noreferrer">bellcheddar/FAFFAbout</a></td>
</tr>
</table>

---

![The FAFFAbout Pipeline Rig showing a forecast for UniProt P0A6Y8, the E. coli chaperone DnaK. Eight gate lamps run from selected through to deposited; the first five are green and crystallised is lit amber as the predicted wall, with cumulative survival falling from 1.00 to 0.18 there. The console on the left carries the sequence, method, centre, host, tag and protease. Below, a precedent table lists real archive targets with censored records greyed and dashed, then counterfactual buttons and the caveats.](docs/screenshots/rig.png)

## 💡 In one minute

Getting from a gene to a protein structure is a long chain of experiments: clone the gene, express the protein, get it soluble, purify it, crystallise it, collect diffraction data, solve the structure, deposit it. Most attempts fail somewhere along the way, and the failures are expensive in time and money.

FAFFAbout forecasts **where along that chain a given protein is most likely to fail**, before any lab work starts. You paste a protein (a FASTA sequence, a UniProt accession or a PDB ID) and it returns:

- the chance of getting through each of the eight steps, and which step is the likeliest wall;
- the real archive proteins most similar to yours, and how far each of them got;
- what changing the expression host or tag would do to the odds;
- a short written explanation, and an honest statement of how much evidence there is.

The forecasts are learned from the **Protein Structure Initiative's TargetTrack archive**: 335,771 proteins attempted by 41 structural genomics centres between 2000 and 2017, with 961,548 experimental trials and 3.8 million recorded status changes. It is the only large record of where protein production attempts *stopped*, not just which ones succeeded.

**Useful for:** triaging a target list before ordering genes, choosing between orthologues, picking a host and tag on evidence rather than habit, and setting realistic expectations with collaborators.

## 🧭 What it is not

It is not a structure predictor and not a replacement for AlphaFold. It predicts *experimental attrition*: whether a real lab attempt gets through each step, which is a different quantity. The archive also has a selection bias: PSI targets were mostly chosen as likely-soluble bacterial proteins, so forecasts for membrane and eukaryotic proteins are extrapolation, and the interface says so.

## ⚙️ How it works: two models, two jobs

FAFFAbout combines two kinds of model, each doing the job it is good at.

| | The GBM | The language model |
|---|---|---|
| What it is | A gradient-boosted model (LightGBM) | Llama 3.1 8B, fine-tuned with LoRA |
| Its job | Every number: the probability of clearing each step | The written explanation of those numbers |
| Why this one | More accurate at this, and well calibrated | Can explain *why*, and cite the precedent |

### What is a GBM?

**GBM stands for gradient-boosted model.** It is a standard, well-understood machine-learning technique for data that comes in rows and columns. It builds hundreds of small decision trees, each one learning to correct the mistakes of the trees before it, and adds their answers together into a single prediction.

In FAFFAbout there is one GBM for each step of the pipeline. Each takes a protein's features (its length, cysteine count, predicted disorder, source organism, and how its relatives in the archive fared) and returns the probability that the protein gets through that step. Chaining the eight steps together gives the forecast: the probability of surviving to each stage, and the **bottleneck**, the step where the most probability is lost.

**Why the numbers come from the GBM and not the language model.** Measured on proteins neither model had seen:

| | GBM | Language model (round 04) |
|---|---|---|
| Ranking successes above failures (AUROC; 0.5 is chance, 1.0 is perfect) | **0.857** | 0.638 |
| Calibration error (lower is better) | **0.024** | 0.036 |
| Names the step where the protein really stopped | **34.6%** | 27.2% |

So the GBM supplies every number on the page, and the language model is trained to do what the GBM cannot: explain the forecast in words a scientist can act on, point to the archive proteins that support it, and say plainly when the evidence is thin.

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

**The language model.** Round 04 invented no identifiers, but it had learned to recite its training wording, named a bottleneck of its own that was right only 27% of the time, and cited a precedent in just 8 of 40 cases. Round 5 was retrained to explain the GBM's forecast instead of inventing one, and it met all five criteria written down before training started:

| Criterion | Round 04 | Round 5 |
|---|---|---|
| Invents a target or PDB identifier | 0 of 40 | **0 of 40** |
| Names the same bottleneck as the GBM | no (27.2% vs 34.6% right) | **yes, on all 235 test proteins** |
| Cites a precedent from the archive | 8 of 40 | **36 of 40** |
| Flags censored precedents when present | 19 of 20 | **29 of 30** |
| Echoes training wording (lower is better; corpus 0.626) | recited | **0.608** |

Whether its explanations add insight a scientist would act on is the question the 40-case expert rubric answers, and that grading is still to do.

## 🖥️ The application

One field takes a FASTA sequence, a UniProt accession or a PDB ID with an optional chain. The resolved sequence is always shown before anything is computed, so a mis-resolved identifier cannot silently produce a confident forecast for the wrong protein.

A request searches all 300,027 distinct archive sequences with MMseqs2 (about 0.8 s), joins the hits to their outcomes and censoring flags, computes the sequence features, and scores each step with its GBM. **Nothing in the browser computes a probability.** The **counterfactual** buttons re-run the forecast with one choice changed (host, tag, protease, codon optimisation, auto-induction), using a second GBM trained only on records that declare a protocol.

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
| GBM forecasts | `scripts/06b_gbm_forecasts.py` | an out-of-fold GBM forecast for every protein, for training prompts |
| Training data | `scripts/07_build_sft.py` | the fine-tuning corpus |
| Fine-tune | `scripts/08_train.py` | a LoRA adapter (run `scripts/preflight.sh` first) |
| Evaluate | `scripts/evaluate_round.sh` | calibration, hallucination check, narrative value, bottleneck accuracy |

Prefix each Python step with `.venv/bin/python`. Tests: `.venv/bin/python -m pytest -q`.

## 🔬 For the technically curious

**The ladder.** All 30 TargetTrack status values map onto nine stages (`config/status_map.yaml`), read from the archive's own controlled vocabulary. A *gate* is the transition out of a stage, so there are eight. NMR and cryo-EM milestones map onto their own ladders. "Expression tested" records that a test ran, not that protein was seen, so it does not advance past cloned. `work stopped` is never a stage or a failure label: its use is a centre convention (JCSG never uses it; CESG closes 93% of its targets with it).

**Labels.** L1 is one row per gate a target actually entered (906,573 rows; 842,681 used in training, the rest censored). L2 is the terminal outcome per uncensored target (271,619, 3.86% deposited). L3 is 172,695 within-family preference pairs, 72,505 of them near-identical proteins (over 70% identity) with different fates.

**Features.** Composition (length, isoelectric point, GRAVY, charge, cysteines), topology (transmembrane helices, signal peptide, disorder by terminus, low complexity), taxonomy, construct choices mined from 1,501 shared protocol documents, and archive context (how the protein's relatives fared), computed over the training split only with each target's own contribution removed. The raw sequence never enters a language-model prompt.

**Splits.** Always by MMseqs2 family (30% identity), never by row, because the same protein appears across many orthologues. The temporal split (train before 2014, test from 2014) is the honest test of forecasting the future and the one the narrative evaluations grade.

**The leak the baseline caught.** The first GBM scored a near-perfect AUROC of 0.985. Three columns were the answer in disguise: a PDB reference exists only for deposited targets, the `method` field is derived from the status history, and successful targets simply have more trials. `config/features.yaml` now separates what a scientist knows *before ordering the gene* from what was recorded *while the attempt ran*, and the leaky configuration is kept as a negative control asserted by a test.

**Training the language model.** Llama 3.1 8B Instruct (8-bit), LoRA rank 16 on all 32 layers, learning rate 2e-5 (the specification's 1e-4 diverged), 2,000 iterations on Apple silicon with MLX. From round 5 each training prompt carries the GBM's forecast, computed out-of-fold so the model never sees the optimistic numbers of a booster scoring a protein it trained on, and each answer explains the forecast's weakest step with reasons specific to that step, a precedent cited from the prompt, and a hedge sized to the evidence. The app sends the model an identical forecast block, and a test checks the two match byte for byte.

**Evaluation.** `scripts/evaluate_round.sh` serves the adapter, verifies by output that the adapter is actually attached, and runs four checks: calibration against the GBM; an automatic fail for any identifier not present in the model's own prompt; *narrative value*, meaning whether the prose tracks the input or recites training text (at 100 cases); and bottleneck accuracy against where proteins really stopped. A 40-case rubric in `eval/generative_review.md` is left for expert grading.

## 🗺️ Status and next steps

- [x] **Data:** archive parsed, normalised, clustered and queryable; censoring derived (19.03%)
- [x] **Labels, features and splits:** leakage-free by construction and checked by tests
- [x] **GBM:** mean AUROC 0.839 on held-out families; ships in the app as the source of every number
- [x] **Application:** built and working locally
- [x] **Language model, round 04:** trained and evaluated; safe on identifiers, but recites its training wording
- [x] **Language model, round 5:** explains the GBM's forecast; meets all five pre-stated criteria
- [ ] **Expert grading** of the 40-case rubric
- [ ] **ESM-2 features** to replace the local disorder predictor
- [ ] **Deploy** to faffabout.mdeller.com
- [ ] **Licence** for the code (the data is CC-BY-SA-4.0; Llama 3.1 has its own licence to check before redistributing a model)

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
