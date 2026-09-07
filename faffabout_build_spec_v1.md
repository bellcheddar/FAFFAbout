# FAFFAbout: Build Specification v1

> **Fine-tuned Attrition Forecasting From Archives.** A structural biologist enters a FASTA, UniProt accession or PDB ID at the start of a new project and receives a full breakdown of the protein plus calibrated guidance on how to proceed, grounded in 330,000 real attempts from the PSI TargetTrack archive.

**Author:** Marc C. Deller, D.Phil. · [marcdeller.com](https://marcdeller.com) · marc@marcdeller.com
**Repo root:** `/Users/dellboy/Documents/Vibe_Coding/faffabout/`
**Target deployment:** `faffabout.mdeller.com` (Flask + gunicorn + nginx, same pattern as AlphaFraud)
**Conventions:** British English. No em dashes (colons or parentheses). Front end uses the Pipeline Rig palette defined in section 9, NOT the marcdeller.com house header.

---

## 1. What this tool is, and what it is not

**It is:** a decision-support tool for the first week of a structure project. You have a sequence. Before you order a gene, FAFFAbout tells you where comparable targets historically died, which construct and host choices moved the needle, and how much confidence the evidence actually supports.

**It is not:** a structure predictor, a fold recogniser, or a replacement for AlphaFold. It predicts *experimental attrition*, which is a different and largely unmodelled quantity.

**The core scientific claim:** the PSI archive is the only large corpus that records where protein production attempts stopped, not just which ones succeeded. Everything downstream depends on extracting that signal without mistaking programme shutdown for scientific failure.

---

## 2. Hardware and model envelope

| Item | Value |
|---|---|
| Machine | Mac Studio / MacBook Pro, Apple M1, 64 GB unified memory |
| Base model | `meta-llama/Llama-3.1-8B-Instruct` (8.03B params, 32 layers, 4096 hidden) |
| Licence | Llama 3.1 Community Licence (not Apache 2.0: check before any redistribution) |
| Quantisation | **8-bit**, not 4-bit. You have the headroom and 8-bit measurably preserves numeric reasoning |
| Fine-tune type | LoRA (QLoRA automatically, since the base is quantised) |
| Adapter layers | All 32 (`num_layers: 32`) |
| Expected peak memory | 14 to 18 GB at `batch_size: 4`, `max_seq_length: 2048`, `grad_checkpoint: true` |
| Expected throughput | 1.5 to 3.0 it/s on M1. **Budget 6 to 10 hours per full training run.** M1 is roughly 2.5x slower than M4 Max for this workload |

**Deviation from the ChemSage recipe, and why.** ChemSage ran 4-bit at `num_layers: 16`, `batch_size: 1`. That configuration was memory-driven. On 64 GB you should spend the headroom: 8-bit base, all 32 layers adapted, `batch_size: 4`. Rank stays at 16. The task here is calibrated numeric judgement over a retrieved evidence table, which degrades under aggressive quantisation in a way that ChemSage's style-and-format task did not.

---

## 3. Repository layout

```
faffabout/
├── PROJECT_PLAN.md
├── README.md
├── requirements.txt
├── config/
│   ├── train_config.yaml          # native mlx_lm.lora config
│   └── features.yaml              # feature extraction toggles
├── data/
│   ├── raw/                       # Zenodo tarball, never committed
│   ├── parquet/                   # normalised tables
│   ├── faffabout.duckdb           # query layer
│   ├── clusters/                  # MMseqs2 output
│   └── sft/                       # train.jsonl, valid.jsonl, test.jsonl
├── scripts/
│   ├── 01_fetch_zenodo.py
│   ├── 02_parse_tt_xml.py
│   ├── 03_normalise_status.py
│   ├── 04_cluster_sequences.sh
│   ├── 05_derive_features.py
│   ├── 06_build_labels.py
│   ├── 07_build_sft.py
│   └── 08_train.py
├── baseline/
│   └── gbm_baseline.py            # the gate the LLM must clear
├── eval/
│   ├── eval_calibration.py
│   └── eval_generative.py
├── app/
│   ├── server.py                  # Flask
│   ├── predict.py                 # inference orchestration
│   ├── retrieve.py                # MMseqs2 + DuckDB retrieval
│   ├── templates/rig.html         # Pipeline Rig (see faffabout_rig_v1.html)
│   └── static/
└── adapters/
    └── faffabout_lora/
```

---

## 4. Phase 1: Parse and normalise the archive

**Goal:** turn one 795 MB tarball into a queryable relational core. **Budget: 1 day.**

### 4.1 Acquire

Zenodo DOI `10.5281/zenodo.821654`, "Protein Structure Initiative: TargetTrack 2000-2017, All Data Files". Final timestamp 30 June 2017. Inside:

- `TargetTrack XML files/tt.xml.gz`: the entire database as one XML document
- `Documentation/targetTrack-v1.4.1.pdf`: the schema
- `Documentation/`: a spreadsheet of TargetTrack status enumerations

**Read the enumerations spreadsheet before writing the parser.** Status vocabulary drifted across TargetDB (2003) to PepcDB (2008) to TargetTrack (2010), and across contributing centres. That spreadsheet is the only authoritative mapping. Do not hand-roll a status list from the PDF prose.

Record the Zenodo licence field in `README.md` before planning any model redistribution.

### 4.2 Parse

Stream, do not DOM-parse. A DOM parse of `tt.xml.gz` will consume tens of GB.

```python
from lxml import etree
context = etree.iterparse(fh, events=("end",), tag="target")
for _, elem in context:
    emit(extract(elem))
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]
```

Write to Parquet in row groups of 50,000 via `pyarrow`. Then attach DuckDB over the Parquet directory. Do not use SQLite here: AlphaFraud's SQLite pattern suits weekly incremental writes, but this is a static 330k-row analytical store and DuckDB will be an order of magnitude faster on the group-bys you need.

**Verify element and attribute names against `targetTrack-v1.4.1.pdf` rather than assuming.** The nesting changed between schema versions and the exact tag names are not reproduced in this spec on purpose.

### 4.3 Normalise into four tables

| Table | Grain | Key columns |
|---|---|---|
| `targets` | one row per target | `target_id`, `centre`, `organism`, `taxon_id`, `sequence`, `seq_md5`, `first_seen`, `last_seen` |
| `status_history` | one row per status event | `target_id`, `status_raw`, `status_canon`, `stage_ord`, `status_date` |
| `trials` | one row per experimental attempt | `target_id`, `trial_id`, `protocol_ref`, `host`, `tag`, `boundaries`, `free_text_notes` |
| `outcomes` | one row per deposition | `target_id`, `pdb_id`, `method`, `resolution`, `deposit_date` |

**Canonical stage ordinal (`stage_ord`), 0 to 8:**

```
0 selected → 1 cloned → 2 expressed → 3 soluble → 4 purified
→ 5 crystallised → 6 diffracting → 7 structure → 8 deposited
```

NMR and cryo-EM targets map gates 5 to 7 onto their own ladders (`labelled`/`assigned`, `vitrified`/`classified`). Keep `method` on the target so the ladder is selectable at inference.

### 4.4 Cluster

```bash
mmseqs easy-cluster targets.fasta clusters/tt30 tmp \
  --min-seq-id 0.30 -c 0.8 --cov-mode 1
```

Every downstream split is by **cluster**, never by row. The archive contains the same protein attempted across five or more orthologues by different centres. A random split leaks catastrophically and will hand you a meaningless 0.9 AUC.

**Phase 1 acceptance test:** `SELECT count(*) FROM targets` returns a number in the low 300,000s, `status_canon` has no NULLs outside a documented unmapped set, and the cluster file resolves every `target_id`.

---

## 5. Phase 2: Construct the positive and negative label set

**This is the phase that determines whether the project works. Budget 2 to 3 days, and do not compress it.**

### 5.1 The censoring problem, stated precisely

A target sitting at `cloned` on 30 June 2017 is not a failure. It is one of three things:

1. **Failed:** tried and abandoned on scientific grounds
2. **Censored:** the centre's funding ended and the file closed with work incomplete
3. **In flight:** still active when the archive froze

Treating all three as negatives will train the model to predict *when a US funding programme ended*, which it will do very well and which is worthless. This is the single largest failure mode of the project.

**Derive censoring empirically per centre:**

```python
# for each centre, find the date after which it effectively stopped reporting
centre_last = status_history.groupby('centre').status_date.quantile(0.995)

censored = (
    (max_stage < 8) &
    ((centre_last[centre] - last_activity).days < 180)
)
```

Also treat the global archive freeze the same way: any target whose last activity falls within 180 days of 2017-06-30 is administratively censored regardless of centre.

Expect roughly 15 to 20% of the archive to fall out as censored. Store `censored` as a first-class boolean column, surfaced in the UI, never silently dropped.

### 5.2 Three label families

Build all three. They serve different training stages.

**L1: gate transitions (the workhorse, supervised).**
For every target and every gate `g` the target actually *entered*, emit one row:

| label | condition |
|---|---|
| `cleared` | the target reached `stage_ord > g` |
| `failed` | the target's max stage is exactly `g` and `censored == False` |
| `censored` | the target's max stage is exactly `g` and `censored == True` |

Rows for gates the target never entered are **not emitted**. Not-attempted is not a negative.

This yields roughly 1.3 to 1.6 million labelled transitions from 330k targets: a large, genuinely balanced-per-gate supervised set. `censored` rows are excluded from the loss but retained in retrieval context.

**L2: terminal outcome (for calibration eval).**
One row per uncensored target: `deposited` or `stalled_at_{g}`. Roughly 4% positive. Use for Brier score and expected calibration error, not for the main SFT loss.

**L3: within-cluster preference pairs (for the optional DPO stage).**
Within each MMseqs2 cluster, for every pair of uncensored targets A and B where `max_stage(A) > max_stage(B) + 1`, emit a preference pair. These are the highest-value examples in the entire corpus: two near-identical proteins, different construct or host, different fate. Cap at 20 pairs per cluster to stop large families dominating.

### 5.3 Negative mining rules

Four rules, all of which need to be in the code and not just the plan.

1. **Censored is never a negative.** Excluded from loss, present in context, flagged in output.
2. **Not-attempted is never a negative.** A target that stopped at `selected` tells you nothing about crystallisation.
3. **Mine hard negatives explicitly.** A same-cluster pair at greater than 70% identity with divergent fate is worth roughly fifty random negatives. Tag these `hard=True` and oversample 3x in the SFT mix.
4. **Downsample trivial negatives.** A 900-residue eukaryotic membrane protein that died at expression is learnable from ten examples. Cap the easy-negative stratum at 25% of the negative mass, stratified on (TM helix count, kingdom, length decile).

### 5.4 Feature extraction: what goes in the prompt

**Never put the raw sequence in the prompt.** Llama's BPE tokeniser shreds amino acid strings at roughly one token per two to three residues with no semantic structure. A 400-residue protein burns about 150 meaningless tokens and teaches the model nothing. This single decision roughly halves training cost.

Compute and serialise instead:

| Group | Features |
|---|---|
| Composition | length, pI, GRAVY, Cys count, Met count, low-complexity fraction, aromatic fraction |
| Topology | predicted TM helix count, signal peptide, coiled-coil fraction, disorder fraction (N-term / C-term / internal) |
| Annotation | Pfam / InterPro domains, EC class if known, organism, kingdom, optimal growth temperature |
| Construct | boundaries, host, tag, protease site, codon optimisation, induction regime, purification route |
| Archive context | precedent count within 30% identity, closest precedent identity, per-cluster historical base rate, censored fraction in cluster |

Disorder and embedding calls run on **ZeroGPU** (`@spaces.GPU`, ESM-2 650M, batched). This is genuinely the right fit: burst inference, sub-60-second calls, and your PRO tier's 40 min/day covers a full corpus pass in a few sessions. Cache every result to Parquet keyed on `seq_md5` so you never pay twice.

### 5.5 Splits

Three, and report all three:

- **Primary:** cluster-held-out, 80/10/10 by MMseqs2 cluster
- **Centre generalisation:** leave-one-centre-out across JCSG, NESG, MCSG, NYSGRC, CESG. Include `centre` as an input feature so the model can condition on pipeline differences rather than silently absorbing them
- **Temporal:** train on pre-2014, test on 2014 onward. This is the honest forecasting test

### 5.6 SFT record format

`data/sft/{train,valid,test}.jsonl`, chat format, one `{"messages": [...]}` per line. Mixed multi-task so the model does not overfit one template. Target mix:

| Task | Share | Output shape |
|---|---|---|
| Gate transition judgement | 45% | probability + one-sentence rationale |
| Full pipeline forecast | 20% | nine-gate probability vector + predicted bottleneck |
| Orthologue ranking | 15% | ranked list with reasoning |
| Construct recommendation | 12% | boundaries/host/tag with justification |
| Failure attribution | 8% | free-text explanation of which features drove pessimism |

**Vary the prompt phrasing.** Generate 8 to 12 paraphrases per task template (use a larger model offline to paraphrase) or the model will learn the template rather than the task. Sort the final JSONL by token length before writing: MLX pads to the longest item in a batch, and variable-length precedent tables otherwise waste 30 to 40% of compute on padding.

**Phase 2 acceptance test:** `censored` fraction is between 12 and 22%; hard-negative pairs number in the tens of thousands; no cluster appears in more than one split; a random sample of 50 SFT records reads as sensible to you as a domain expert.

---

## 6. Phase 3: Baseline, then fine-tune

**Budget: 2 days plus overnight training runs.**

### 6.1 Build the gradient-boosted baseline first. Non-negotiable.

`baseline/gbm_baseline.py`: XGBoost or LightGBM on the exact same tabular features, predicting L2 terminal outcome and per-gate L1 transitions.

**Expect the GBM to beat the LLM on AUC.** That is the normal result for tabular prediction and it does not kill the project. What it does is define what the LLM is for: the narrative, the attribution, the construct recommendation, the orthologue ranking. If you cannot state a number the LLM adds over the GBM, you are shipping decoration.

**Ship the GBM inside the app** as the source of the probability numbers, and let the fine-tuned model do the reasoning over them. That hybrid is more honest and more accurate than asking an 8B model to emit calibrated floats.

### 6.2 MLX-LM configuration

`config/train_config.yaml`, native `mlx_lm.lora` format:

```yaml
# FAFFAbout LoRA fine-tune (MLX-LM native)
#   mlx_lm.lora --config config/train_config.yaml --train
#
# Field names drift between mlx-lm versions. Before the first run:
#   curl -O https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/examples/lora_config.yaml
# and reconcile keys against your installed version.

# --- Model ---
# Quantise once:  mlx_lm.convert --hf-path meta-llama/Llama-3.1-8B-Instruct -q --q-bits 8
model: "./models/Llama-3.1-8B-Instruct-8bit"

train: true
fine_tune_type: "lora"

# --- Data ---
data: "data/sft"

# --- Adapter output ---
adapter_path: "adapters/faffabout_lora"

# --- LoRA ---
num_layers: 32                  # all layers: this task is far from the base distribution
lora_parameters:
  keys: ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
  rank: 16
  scale: 16.0                   # MLX scale, not identical to HF alpha
  dropout: 0.05                 # nonzero: the SFT set has templated structure

# --- Optimisation ---
iters: 6000                     # approx (n_examples / batch_size) * 2 epochs
batch_size: 4
learning_rate: 1.0e-4           # half the ChemSage rate: 32 adapted layers, not 16
lr_schedule:
  name: cosine_decay
  warmup: 200
  arguments: [1.0e-4, 6000, 1.0e-6]
max_seq_length: 2048
grad_checkpoint: true
mask_prompt: true               # essential: long retrieved context, short completion
seed: 42

# --- Reporting ---
steps_per_report: 25
steps_per_eval: 250
val_batches: 40
save_every: 500
```

`mask_prompt: true` matters more here than it did for ChemSage. Prompts carry long precedent tables and completions are short, so unmasked loss would mostly teach the model to echo its own context.

### 6.3 Optional preference stage

If the SFT model is overconfident on sparse target classes (it will be), run a second pass on the L3 pairs with `mlx-lm-lora` (Goekdeniz-Guelmez), which adds DPO and ORPO on the same MLX backend:

```bash
mlx_lm_lora.train --model ./models/Llama-3.1-8B-Instruct-8bit \
  --train --data data/dpo --train-mode dpo --train-type lora \
  --adapter-path adapters/faffabout_dpo --iters 1200
```

Chosen = calibrated hedge with censoring flagged. Rejected = confident assertion. This is what teaches the model to say "the evidence here is thin" rather than inventing a number.

### 6.4 Fuse and export

```bash
mlx_lm.fuse --model ./models/Llama-3.1-8B-Instruct-8bit \
  --adapter-path adapters/faffabout_lora \
  --save-path ./models/faffabout-8b --de-quantize
```

`--de-quantize` is required after QLoRA. Because the base is Llama, MLX's GGUF export path is available to you (it does not support Qwen), so Ollama serving is an option this time. Otherwise serve via `mlx_lm.server` on `127.0.0.1:8080` as with ChemSage.

### 6.5 Evaluation

`eval/eval_calibration.py` reports, on all three splits:

| Metric | Why |
|---|---|
| Brier score | the headline: is the probability honest |
| Expected calibration error | 10-bin, the number to quote |
| AUROC per gate | comparability with the GBM |
| Bottleneck top-1 accuracy | does it name the right wall |
| GBM delta | the number that justifies the fine-tune |

`eval/eval_generative.py`: 40 held-out cases graded by you against a rubric (correct bottleneck, cited real precedents, no invented target or PDB IDs, censoring flagged where present). **Hallucinated target IDs are an automatic fail and mean facts have leaked into the weights that should have stayed in retrieval.**

---

## 7. Phase 4: Serve

**Budget: 2 days.** The front end already exists as `faffabout_rig_v1.html`: port it to a Jinja template rather than rebuilding.

### 7.1 Architecture

```
nginx → gunicorn → Flask (app/server.py)
                     ├── retrieve.py  → MMseqs2 search + DuckDB (Parquet)
                     ├── features.py  → local calcs + ZeroGPU (ESM-2, cached)
                     ├── gbm.py       → probability vector
                     └── llm.py       → mlx_lm.server on 127.0.0.1:8080
```

Single droplet, same shape as AlphaFraud. The DuckDB file and Parquet directory are read-only at runtime.

### 7.2 Input contract

One field accepts all three: raw FASTA, UniProt accession (`P0A6Y8`), or PDB ID with optional chain (`4XB7_A`). Sniff the format, resolve accessions and PDB IDs to sequence via the EBI Proteins and RCSB APIs, and always show the user the resolved sequence before computing.

### 7.3 Output contract (`POST /predict` returns)

```json
{
  "target": {"source": "uniprot", "accession": "...", "length": 312, "organism": "..."},
  "features": {"pI": 5.8, "gravy": -0.21, "disorder_nterm": 0.11, "tm_helices": 0, "pfam": ["PF00561"]},
  "ladder": ["selected", "cloned", "expressed", "soluble", "purified",
             "crystallised", "diffracting", "structure", "deposited"],
  "survival": [1.0, 0.94, 0.81, 0.52, 0.44, 0.19, 0.11, 0.08, 0.07],
  "conditional": [1.0, 0.94, 0.86, 0.64, 0.85, 0.43, 0.58, 0.73, 0.88],
  "bottleneck": {"gate": 3, "name": "soluble", "drop": 0.29,
                 "interval": [0.38, 0.64]},
  "evidence": {"n_precedents": 5, "n_censored": 2, "closest_identity": 0.71,
               "strength": "MODERATE"},
  "precedents": [{"target_id": "...", "centre": "...", "organism": "...",
                  "identity": 0.71, "max_stage": 4, "censored": false,
                  "note": "..."}],
  "counterfactuals": [{"change": "host:arctic", "delta_pts": 11,
                       "moves_bottleneck_to": "crystallised"}],
  "narrative": "…generated by the fine-tuned model…",
  "caveats": ["2 of 5 precedents censored at centre closure",
              "no precedent above 80% identity"]
}
```

The `narrative` field is the only one the LLM writes. Every number comes from the GBM or from DuckDB. **If the model ever emits a target ID, PDB ID or numeric probability, that is a bug.**

### 7.4 Front end

Pipeline Rig as the main view, Necropolis archive map as the secondary, mode selector rather than a tab bar. Palette:

```
--steel-900 #0d0f11   --steel-600 #191c1f   --steel-400 #2c3237
--ink-100   #f2f5f7   --ink-300   #9aa4ab   --ink-500   #4b545a
--lamp-amber #ffb020  --lamp-green #4fd18b  --lamp-red #e07070  --lamp-grey #8a8aa0
```

Typography: Barlow Condensed (UI), IBM Plex Mono (data), Inter (prose). No marcdeller.com brand header on this app.

**Censoring must be visible in the interface at all times**, not buried in a caveats array. Censored precedents render grey and dashed, never red.

---

## 8. Phase summary

| Phase | Deliverable | Budget | Gate to pass |
|---|---|---|---|
| 1 | DuckDB + Parquet core, MMseqs2 clusters | 1 day | 330k targets queryable, statuses canonicalised |
| 2 | L1/L2/L3 label sets, features, splits | 2 to 3 days | Censoring rate 12 to 22%, no cluster leakage, 50 records pass expert eyeball |
| 3 | GBM baseline + LoRA adapter + eval | 2 days plus overnight runs | LLM narrative adds something the GBM cannot, zero hallucinated IDs |
| 4 | Flask app on faffabout.mdeller.com | 2 days | End-to-end FASTA in, full breakdown out |

**Total: roughly 8 working days plus two overnight training runs.**

---

## 9. Known risks

| Risk | Mitigation |
|---|---|
| Model learns "when did US funding end" | Phase 2.1 censoring. The highest-priority correctness issue in the project |
| Homology leakage inflates metrics | Cluster-level splits only, at 30% identity |
| Centre effects dominate | `centre` as an explicit feature plus leave-one-centre-out eval |
| Selection bias in the archive | Document it. PSI targets already passed a soluble-prokaryotic filter, so membrane and eukaryotic predictions are extrapolation. Say so in the UI |
| Hallucinated precedent IDs | Never fine-tune retrievable facts. Automatic eval fail |
| M1 training slower than expected | Drop to `num_layers: 16` and `batch_size: 2` before dropping to 4-bit |

---

## 10. First task for Claude Code

**Scope strictly to Phase 1.** Do not begin label construction, feature extraction, model work or front-end work in this pass.

1. Scaffold the repository as laid out in section 3, with `requirements.txt` and `.gitignore` (exclude `data/raw/`, `data/parquet/`, `adapters/`, `*.safetensors`, `*.gguf`).
2. Write `scripts/01_fetch_zenodo.py`: download and verify the tarball from DOI `10.5281/zenodo.821654`, extract to `data/raw/`, and print an inventory of the `Documentation/` folder.
3. Write `scripts/02_parse_tt_xml.py` using `lxml.iterparse` with the memory-safe clearing pattern, emitting the four Parquet tables in section 4.3. **Leave the element-name mapping as a clearly marked TODO block:** it must be filled in against `targetTrack-v1.4.1.pdf` after inspection, not guessed.
4. Write `scripts/03_normalise_status.py` reading the enumerations spreadsheet and producing the `status_canon` / `stage_ord` mapping, with an explicit unmapped-status report.
5. Write `scripts/04_cluster_sequences.sh` for the MMseqs2 call in section 4.4.
6. Produce `PROJECT_PLAN.md` covering all four phases, and a `README.md` to the house standard (emoji H1, tagline blockquote, shields.io badge row ending in the navy author badge, contact table, "Why it matters:" intro paragraph, tables for flags and outputs, British English, no em dashes).

Report back with the Phase 1 acceptance test results before proceeding to Phase 2.

---

*Built by **Marc C. Deller, D.Phil.** · [marcdeller.com](https://marcdeller.com) · marc@marcdeller.com*
