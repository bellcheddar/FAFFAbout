#!/usr/bin/env python
"""07_build_sft.py: the supervised fine-tuning corpus.

Five task types in the specification's mix, chat JSONL, one {"messages": [...]} per line.

  gate_judgement       45%   a verdict in words on gate g, given the GBM's estimate for it
  pipeline_forecast    20%   narrate the GBM's forecast: its weakest point, and why
  orthologue_ranking   15%   rank the given precedents by likely success, with reasoning
  construct_recommend  12%   boundaries, host and tag, aimed at the GBM's weakest step
  failure_attribution   8%   explain a stop that the prompt states

From round 5 (2026-09-27) the model narrates and never forecasts: every prompt that needs
numbers carries the GBM's, out-of-fold for training targets (scripts/06b_gbm_forecasts.py),
and no completion contains a probability. Reasons are specific to the gate in question
(gate_cues) and every completion with a precedent table cites one of its rows.

Three rules from spec section 5.3 are enforced here rather than described:

  * censored rows never enter the corpus as negatives (they are dropped from L1's loss
    upstream, and this script asserts none survive)
  * hard negatives (same 70%-identity cluster, divergent fate) are oversampled 3x
  * the easy-negative stratum is capped at 25% of the negative mass, stratified on
    (TM helix count, kingdom, length decile)

The raw sequence never appears in a prompt. Retrieved precedents carry real target IDs so
the model learns to cite what it was given rather than to invent identifiers; nothing in a
completion names an ID that was not in its own prompt, which `eval_generative.py` checks.

Prompt phrasing is drawn from 8 to 12 paraphrases per template so the model learns the
task rather than the template. Records are sorted by token length before writing, because
MLX pads to the longest item in a batch and variable-length precedent tables otherwise
waste 30 to 40% of compute on padding.

Usage
  .venv/bin/python scripts/07_build_sft.py
  .venv/bin/python scripts/07_build_sft.py --max-records 200000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "faffabout.duckdb"
PQ = ROOT / "data" / "parquet"
OUT = PQ.parent / "sft"
CFG = ROOT / "config" / "labels.yaml"

LADDER = ["selected", "cloned", "expressed", "soluble", "purified",
          "crystallised", "diffracting", "structure", "deposited"]
GATE_DESC = [
    "cloning the gene into an expression vector",
    "getting detectable expression",
    "getting soluble protein",
    "purifying it to homogeneity",
    "obtaining crystals",
    "obtaining useful diffraction",
    "solving the structure",
    "depositing the structure",
]

SYSTEM = (
    "You are FAFFAbout, an attrition forecaster for structural biology. You judge where a "
    "protein production attempt is likely to stop, using the Protein Structure Initiative "
    "TargetTrack archive of 335,771 real attempts as precedent. You are careful about "
    "censored records, which are targets whose file closed when a centre's funding ended "
    "rather than because the science failed: they are evidence of nothing and you say so. "
    "You never invent a target identifier, a PDB code or a precedent that is not in the "
    "context you were given."
)

# 8 to 12 paraphrases per task, so the model learns the task rather than the template.
PARAPHRASES = {
    # Phrased around the ACTION, because naming a gate after the stage it leaves and then
    # describing what it achieves reads as a contradiction ("whether cloned will work
    # (getting detectable expression)").
    "gate_judgement": [
        "Will this target get as far as {action}?",
        "Given the evidence below, is {action} likely to succeed here?",
        "Assess the chance of {action} for this construct.",
        "How likely is this protein to get through {action}?",
        "Judge whether {action} will work for this protein.",
        "What are the prospects for {action} with this target?",
        "Is {action} going to be the wall for this one?",
        "Estimate the probability of {action} for this target.",
        "Should I expect {action} to succeed for this protein?",
        "Rate the likelihood of {action}.",
    ],
    "pipeline_forecast": [
        "Forecast the whole pipeline for this target.",
        "Where is this protein most likely to stall, and how far will it get?",
        "Give me the full gate-by-gate outlook for this target.",
        "Run the complete attrition forecast on this protein.",
        "How far through the pipeline will this target get, and where does it stop?",
        "Predict this target's fate at every stage.",
        "What does the whole pipeline look like for this one?",
        "Give a stage-by-stage forecast and name the bottleneck.",
        "Assess this target from selection through to deposition.",
        "Which step is going to kill this project?",
    ],
    "orthologue_ranking": [
        "Rank these orthologues by which is most likely to yield a structure.",
        "Which of these candidates should I put first?",
        "Order these homologues from best to worst prospect.",
        "I can only take two of these forward. Which, and why?",
        "Sort these precedents by likely success.",
        "Which of these related targets is the safest bet?",
        "Prioritise this list of orthologues for a structure campaign.",
        "Given these options, what is the order of attack?",
        "Rank these by prospect and justify the top choice.",
        "Which orthologue would you start with?",
    ],
    "construct_recommend": [
        "Recommend a construct and expression strategy for this target.",
        "What boundaries, host and tag would you use here?",
        "How should I design this construct?",
        "Suggest a construct, host and tag, with reasons.",
        "What is the best starting construct for this protein?",
        "Advise on boundaries and expression conditions for this target.",
        "Design the first construct for this protein.",
        "What would you try first for expression and purification here?",
        "Give me a construct plan for this target.",
        "Which host and tag does the precedent support for this one?",
    ],
    "failure_attribution": [
        "Why is the outlook for this target poor?",
        "Explain what is driving the pessimism here.",
        "Which features make this a difficult target?",
        "What is wrong with this protein from a tractability point of view?",
        "Account for the low prospects of this target.",
        "Talk me through the risk factors for this one.",
        "What about this protein worries you?",
        "Diagnose the difficulty with this target.",
        "Which properties are the problem here?",
        "Why would this target be expected to stall?",
    ],
}


def fmt_features(r: dict) -> str:
    """The feature block that stands in for the sequence. Never the residues themselves."""
    bits = [
        f"length: {r['seq_len']} residues" if r.get("seq_len") else None,
        f"organism: {clean_organism(r.get('organism'))}" if clean_organism(r.get("organism")) else None,
        f"kingdom: {r['superkingdom']}" if present(r.get("superkingdom")) else None,
        f"centre: {r['centre']}" if present(r.get("centre")) else None,
        f"pI: {r['pi']:.2f}" if present(r.get("pi")) else None,
        f"GRAVY: {r['gravy']:.3f}" if present(r.get("gravy")) else None,
        f"net charge at pH 7: {r['net_charge_ph7']:+.1f}" if present(r.get("net_charge_ph7")) else None,
        f"cysteines: {r['cys_count']}" if present(r.get("cys_count")) else None,
        f"predicted TM helices: {r['tm_helices']}" if present(r.get("tm_helices")) else None,
        "signal peptide predicted" if r.get("signal_peptide") else None,
        # All three disorder fields are interpolated, so all three must be present: guarding
        # on disorder_frac alone let a NaN N- or C-terminal value through as "nan%".
        f"disorder: {r['disorder_frac']:.0%} overall, {r['disorder_nterm']:.0%} N-terminal, "
        f"{r['disorder_cterm']:.0%} C-terminal" if all(
            present(r.get(k)) for k in ("disorder_frac", "disorder_nterm", "disorder_cterm")) else None,
        f"low-complexity: {r['low_complexity_frac']:.0%}" if present(r.get("low_complexity_frac")) else None,
        f"archive annotation: {r['protein_types']}" if present(r.get("protein_types")) else None,
        f"construct type: {r['target_construct_type']}" if present(r.get("target_construct_type")) else None,
        f"host: {r['host']}" if present(r.get("host")) else None,
        f"tag: {r['tag']}" if present(r.get("tag")) else None,
    ]
    return "\n".join(f"  {b}" for b in bits if b)


def present(v) -> bool:
    """Is this value actually there?

    `float('nan')` is TRUTHY and is not None, so it defeats both `if r.get('x')` and
    `if r.get('x') is not None`. Every guard in this file used one of those, and pandas
    hands NaN back for any missing numeric or string column, so the absent values sailed
    straight into f-strings. `:.2f` and `:.0%` render NaN as "nan" rather than raising.

    The result, measured on 2026-09-16: 10,391 TRAINING TARGETS (8.7%) contained the literal
    string, teaching the model to write it as though it were a protocol choice:

        Tag: nan, cleaved with nan.          8,304 targets
        Host: nan, following the precedent.  4,609 targets

    This is the same shape as the DNA-in-the-protein-column bug: a value that looks present,
    is not, and passes every check that asks "is it there?" instead of "is it valid?".
    """
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return str(v).strip().casefold() not in ("", "nan", "none", "<na>")


JUNK_ORGANISM = {"", "other", "unknown", "unidentified", "n/a", "na", "none", "synthetic construct"}


def clean_organism(o) -> str:
    """The archive writes 'Other' and 'unknown' as organism names; printing them as if
    they were species reads as a hallucination."""
    s = (o or "").strip()
    return "" if s.casefold() in JUNK_ORGANISM else s


def fmt_precedents(rows: list[dict]) -> str:
    if not rows:
        return "  (no precedent within 30% identity)"
    # Identifiers are NEVER truncated. Printing "NYCOMPS-GO.78" in the prompt while the
    # completion says "NYCOMPS-GO.7810" teaches the model to extend an identifier it was
    # given, which is the exact habit behind hallucinated precedent IDs.
    out = ["  target              centre    organism                        reached          note"]
    for p in rows[:8]:
        note = "censored at centre closure" if p.get("censored") else ""
        stage = LADDER[p["max_stage"]] if p.get("max_stage") is not None else "unknown"
        out.append(f"  {p['target_id']:<19} {p['centre']:<9} "
                   f"{clean_organism(p.get('organism')):<31} {stage:<16} {note}".rstrip())
    return "\n".join(out)


def fmt_evidence(r: dict) -> str:
    n = r.get("n_precedents") or 0
    close = r.get("n_close_precedents") or 0
    cens = r.get("cluster_censored_frac")
    strength = "STRONG" if n >= 20 and close >= 3 else "MODERATE" if n >= 5 else "WEAK"
    line = f"  {n} uncensored precedents in the 30% cluster, {close} above 70% identity"
    if cens is not None:
        # Omit rather than render. cluster_censored_frac is a ratio over TRAINING-split
        # members of the cluster (05_derive_features.py:323), and the 30% identity split
        # makes held-out clusters disjoint from training ones, so n_train = 0 and the CASE
        # returns NULL for every held-out row. That is the leave-one-out discipline working,
        # not missing data: there is genuinely no training precedent to average. Printing it
        # as "nan% of this cluster's records are censored" put a broken-looking statistic in
        # front of the model in 100% of valid and test prompts.
        line += f"\n  {cens:.0%} of this cluster's records are censored" if present(cens) else ""
    return f"{line}\n  evidence strength: {strength}"


SHRINKAGE = 5.0   # pseudo-counts pulling a thin cluster back towards the global gate rate


def calibrated_p(r: dict, priors: dict[int, float]) -> float:
    """The evidence-based probability, never a function of this target's own outcome.

    Training the completion to say 0.85 for every success and 0.15 for every failure
    teaches confident guessing, which is the overconfidence the spec warns about. A
    single target's fate is one Bernoulli draw; the honest target is the base rate for
    its bucket, shrunk towards the global gate rate when the cluster is thin.
    """
    g = r["gate"]
    prior = priors.get(g, 0.5)
    n = r.get("n_precedents") or 0
    k = r.get("n_precedents_cleared")
    if k is None and r.get("cluster_base_rate") is not None:
        k = (r["cluster_base_rate"] or 0) * n
    k = k or 0
    p = (k + SHRINKAGE * prior) / (n + SHRINKAGE)
    # features that genuinely move the odds, in the direction the archive supports
    if (r.get("tm_helices") or 0) >= 3 and g >= 2:
        p *= 0.75
    if (r.get("disorder_frac") or 0) > 0.4 and g >= 4:
        p *= 0.80
    if (r.get("seq_len") or 0) > 600 and g >= 3:
        p *= 0.85
    if r.get("superkingdom") == "Eukaryota" and g >= 1:
        p *= 0.85
    return min(0.97, max(0.03, p))


VERDICTS = [  # (lower bound on the GBM's conditional, phrasings)
    (0.75, ["Likely to clear.", "This step should go through.", "Good prospects at this step."]),
    (0.50, ["More likely than not to clear, but not comfortably.", "Probably clears, with some risk.",
            "Leaning towards success here."]),
    (0.30, ["A real risk of stalling here.", "Doubtful.", "This step could well be the wall."]),
    (0.00, ["Unlikely to clear.", "This is where it is most likely to stop.", "Poor prospects at this step."]),
]


def gate_judgement(r: dict, rng: random.Random, priors: dict[int, float]) -> dict | None:
    """Round 5: the prompt carries the GBM's estimate for this gate and the answer is a
    verdict in words plus reasons, never a probability of its own."""
    g = r["gate"]
    cond = r.get("gbm")
    if not cond or cond[g] is None or cond[g] != cond[g]:
        return None
    p = cond[g]
    q = rng.choice(PARAPHRASES["gate_judgement"]).format(action=GATE_DESC[g])
    user = (f"{q}\n\nTarget features:\n{fmt_features(r)}\n\n"
            f"Gate under consideration: {LADDER[g]} -> {LADDER[g+1]} ({GATE_DESC[g]})\n\n"
            f"Archive evidence:\n{fmt_evidence(r)}"
            + (f"\n\nPrecedents:\n{fmt_precedents(r['precedents'])}" if r.get("precedents") else "")
            + f"\n\nModel estimate for this gate (gradient-boosted model; fixed): {p:.2f}")
    verdict = rng.choice(next(v for lo, v in VERDICTS if p >= lo))
    assistant = f"{verdict} {narrative(r, g, rng)}"
    return {"task": "gate_judgement", "target_id": r["target_id"], "gate": g,
            "label": r.get("label"), "p": round(p, 4), "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}]}


def oxford(items: list[str]) -> str:
    """Join a list the way a person writes one.

    `", ".join(cues[:3])` produced 14,447 targets reading "carries substantial length, a
    high cysteine count." with no conjunction, which is not English and is exactly the kind
    of thing a model learns to reproduce.
    """
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def evidence_is_positive(r: dict) -> bool:
    """Tone from the CLUSTER's record, never from this target's own outcome.

    build_reason used to be called with `reached >= 5` (pipeline_forecast) and
    `max_stage >= 4` (construct_recommend), so the prose said "comparable targets in this
    cluster generally get through" precisely when THIS target got through. That is the
    answer written into the explanation, and at inference there is no answer to read.
    """
    n = r.get("n_precedents") or 0
    k = r.get("n_precedents_cleared")
    if k is not None and n:
        return (k / n) >= 0.5
    base = r.get("cluster_base_rate")
    return bool(present(base) and base >= 0.5)


def censoring_caveat(r: dict) -> str:
    """Spec rule 1: censored is excluded from the loss, present in context, and FLAGGED IN
    OUTPUT. Only the ranking task said so, and the generative eval scored 0 of 19 cases
    that had a censored precedent in context."""
    cens = [p for p in (r.get("precedents") or []) if p.get("censored")]
    frac = r.get("cluster_censored_frac") or 0
    if cens:
        n, tot = len(cens), len(r.get("precedents") or [])
        # "of the N shown", not "here": the prompt quotes CLUSTER-wide counts while this
        # sentence counts only the capped list of precedents displayed. Reporting two
        # populations in one paragraph without saying so made 44,872 targets look as though
        # they contradicted their own evidence block. The verb agrees, too: 18,201 targets
        # read "1 of N precedents here are censored".
        verb = "is" if n == 1 else "are"
        return (f" Note that {n} of the {tot} precedents shown above {verb} censored: their "
                "files closed when the centre stopped reporting, so they are evidence of "
                "nothing either way and the real base is thinner than the count suggests.")
    if frac >= 0.05:
        # a censored share of the cluster is context worth commenting on even when no
        # censored row made it into the eight precedents shown
        return (f" Bear in mind that {frac:.0%} of this cluster's records are censored at "
                "centre closure rather than by any experimental result, so the effective "
                "evidence is thinner than the counts suggest.")
    return ""


def build_reason(r: dict, cleared: bool) -> str:
    """One sentence grounded in the features that are actually present."""
    cues = []
    if (r.get("tm_helices") or 0) >= 3:
        cues.append("multiple predicted transmembrane helices")
    elif (r.get("tm_helices") or 0) >= 1:
        cues.append("a predicted membrane anchor")
    if (r.get("disorder_frac") or 0) > 0.4:
        cues.append("extensive predicted disorder")
    elif (r.get("disorder_nterm") or 0) > 0.5:
        cues.append("a disordered N-terminus that would be worth trimming")
    if (r.get("seq_len") or 0) > 600:
        cues.append("substantial length")
    if (r.get("low_complexity_frac") or 0) > 0.2:
        cues.append("low-complexity segments")
    if r.get("superkingdom") == "Eukaryota":
        cues.append("a eukaryotic source, which this archive handles poorly")
    if (r.get("cys_count") or 0) > 8:
        cues.append("a high cysteine count")
    n = r.get("n_precedents") or 0
    if cleared:
        head = "Comparable targets in this cluster generally get through"
        if n < 5:
            head = "The precedent is thin, but what there is points the right way"
        tail = f" despite {oxford(cues[:2])}" if cues else ""
        return f"{head}{tail}."
    head = "The precedent in this cluster is discouraging" if n >= 5 else \
           "There is little precedent here, and the intrinsic properties are unhelpful"
    tail = f", and this target carries {oxford(cues[:3])}" if cues else ""
    return f"{head}{tail}."


# --- round 5: the model narrates the GBM's forecast ------------------------------------
#
# Round 04 was trained to INVENT the gate vector and name the lowest-conditional gate. That
# broke CLAUDE.md rule 5 (every number comes from the GBM), the app discarded the numbers
# anyway (app/llm.py strip_generated_numbers), and the rule it learned named the true wall
# 24.0% of the time against 22.8% for always guessing the commonest wall. The GBM under the
# app's own largest-drop rule gets 34.6% (eval/eval_bottleneck.py). So from round 5 the
# prompt carries the GBM's forecast (out-of-fold for training targets, from
# scripts/06b_gbm_forecasts.py), the weakest point is the app's largest-drop gate, and the
# completion is prose: why THIS target is weak at THAT gate, which precedent says so, and
# how far to trust it. No probabilities: the GBM's are shown beside the prose.

def forecast_drop_gate(cond: list[float]) -> int:
    """app/predict.py's bottleneck(): the gate that loses the most survival mass."""
    surv, run = [1.0], 1.0
    for c in cond:
        run *= c
        surv.append(run)
    return max(range(8), key=lambda i: surv[i] - surv[i + 1])


def fmt_forecast(cond: list[float]) -> str:
    """The GBM block, in the exact shape app/llm.py builds from the /predict payload."""
    lines, run = [], 1.0
    for g, c in enumerate(cond):
        run *= c
        lines.append(f"  {LADDER[g]} -> {LADDER[g+1]}: {c:.2f} conditional, {run:.2f} cumulative")
    g = forecast_drop_gate(cond)
    return ("Model forecast (gradient-boosted model; these numbers are fixed):\n" + "\n".join(lines)
            + f"\n  weakest point: {LADDER[g]} -> {LADDER[g+1]} (largest loss of survival)")


def gate_cues(r: dict, g: int) -> list[str]:
    """Features that bear on THIS gate, in this target, worded for that gate.

    build_reason lists the same cues whatever the gate, so a disordered C-terminus was
    offered as the reason cloning would fail. A cue belongs to the gates where the
    structural-biology mechanism actually acts.
    """
    tm, dis = r.get("tm_helices") or 0, r.get("disorder_frac") or 0
    dn, dc = r.get("disorder_nterm") or 0, r.get("disorder_cterm") or 0
    n, cys = r.get("seq_len") or 0, r.get("cys_count") or 0
    lc, gravy = r.get("low_complexity_frac") or 0, r.get("gravy")
    euk = r.get("superkingdom") == "Eukaryota"
    cues: list[str] = []
    if g == 0:
        if n > 1000: cues.append(f"a {n}-residue open reading frame is a long gene to clone intact")
        if lc > 0.2: cues.append("low-complexity, repetitive sequence tends to be awkward to amplify and assemble")
    if g == 1:
        if euk: cues.append("a eukaryotic protein often expresses poorly in the bacterial hosts this archive relied on")
        if tm >= 3: cues.append(f"{tm} predicted transmembrane helices put a heavy load on the host's membrane insertion machinery")
        elif tm >= 1: cues.append("a predicted membrane anchor can make expression toxic or low")
        if r.get("signal_peptide"): cues.append("a predicted signal peptide targets the protein away from the cytoplasm")
        if n > 800: cues.append(f"at {n} residues it is large for routine bacterial expression")
    if g == 2:
        if tm >= 1: cues.append("membrane segments usually leave the protein in the insoluble fraction without detergent")
        if cys > 8: cues.append(f"{cys} cysteines raise the risk of misfolding and aggregation in a reducing cytoplasm")
        if present(gravy) and gravy > 0: cues.append("an overall hydrophobic composition favours aggregation")
        if dis > 0.4: cues.append("extensive predicted disorder makes a stable, soluble fold less likely")
    if g == 3:
        if dis > 0.4: cues.append("extensive predicted disorder invites proteolysis and heterogeneity during purification")
        if n > 600: cues.append("a long chain is harder to purify intact")
        if tm >= 1: cues.append("any membrane segment drags detergent into every purification step")
    if g == 4:
        if dis > 0.3: cues.append("a large disordered fraction is the classic obstacle to crystal packing")
        if dn > 0.5: cues.append("a disordered N-terminus would be worth trimming before crystallisation trials")
        if dc > 0.5: cues.append("a disordered C-terminus would be worth trimming before crystallisation trials")
        if lc > 0.15: cues.append("low-complexity segments add conformational heterogeneity")
        if tm >= 3: cues.append("polytopic membrane proteins rarely crystallise without extensive screening")
    if g == 5:
        if dis > 0.3: cues.append("residual flexibility tends to limit diffraction even when crystals form")
        if n > 600: cues.append("large, multi-domain proteins often give crystals that diffract poorly")
    if g >= 6 and not cues:
        cues.append("late-stage losses in this archive mostly reflect phasing and refinement effort rather than the sequence")
    return cues


LEADS = {
    "weak": ["The forecast puts the main risk at {act}.",
             "The weakest point in this forecast is {act}.",
             "Most of the attrition in this forecast falls at {act}.",
             "If this target stalls, the forecast says it will most likely be at {act}."],
    "because": ["For this protein, {cues}.", "Here, {cues}.", "In this target's case, {cues}.",
                "Looking at the features, {cues}."],
    "nocue": ["Nothing in the sequence features points to that step in particular, so the forecast is carried by the archive's record for comparable targets.",
              "The target's own properties are unremarkable for that step; the forecast rests on how comparable targets fared.",
              "No single feature of this protein explains that step, so the risk comes from the archive's record rather than the sequence."],
    # failure_attribution explains a stop that already happened: there is no forecast to cite
    "stopped": ["The stop came at {act}.", "It failed at {act}.", "The attempt ended at {act}.",
                "What went wrong was {act}."],
    "nocue_explain": ["Nothing in its sequence features explains that stop, so it most likely reflects how hard this family is in general.",
                      "Its own properties are unremarkable for that step, which points to the family rather than this sequence.",
                      "No single feature of this protein accounts for it; the difficulty looks like the family's."],
}


def precedent_sentence(r: dict, g: int, rng: random.Random) -> str:
    """Cite a precedent FROM THE PROMPT that bears on this gate, or say there is none.

    Round 04 cited a precedent in 8 of 40 evaluated cases, because only the orthologue task
    ever named one. Every identifier here comes from r['precedents'], which is exactly the
    table printed in the prompt, so the habit taught is citing what was given.
    """
    shown = (r.get("precedents") or [])[:8]
    if not shown:
        return rng.choice(["There is no precedent within 30% identity, so this rests on the protein's own properties.",
                           "No relative within 30% identity is in the archive, which leaves the target's own features to carry the forecast."])
    live = [p for p in shown if not p.get("censored") and p.get("max_stage") is not None]
    if not live:
        return (f"The {len(shown)} precedents shown are all censored at centre closure, so they say "
                "nothing about this step either way.")
    same = [p for p in live if p["max_stage"] == g]
    past = [p for p in live if p["max_stage"] > g]
    if same:
        p = rng.choice(same)
        return rng.choice([f"{p['target_id']} stopped at exactly this point, at {LADDER[g]}.",
                           f"The precedent agrees: {p['target_id']} got no further than {LADDER[g]}."])
    if past:
        p = max(past, key=lambda q: q["max_stage"])
        return rng.choice([f"{p['target_id']} got past this point and reached {LADDER[p['max_stage']]}, so it is not a hard wall for this family.",
                           f"It is not insurmountable: {p['target_id']} reached {LADDER[p['max_stage']]}."])
    p = max(live, key=lambda q: q["max_stage"])
    return (f"None of the uncensored precedents reached {LADDER[g]}; the furthest, {p['target_id']}, "
            f"stopped at {LADDER[p['max_stage']]}.")


def hedge(r: dict, rng: random.Random) -> str:
    n = r.get("n_precedents") or 0
    close = r.get("n_close_precedents") or 0
    if n >= 20 and close >= 3:
        return rng.choice([" The evidence here is strong enough to take seriously.",
                           " With this much close precedent, the forecast is well grounded."])
    if n >= 5:
        return rng.choice([" Treat this as a reasonable guide rather than a firm prediction.",
                           " The evidence is moderate, so hold the conclusion loosely."])
    return rng.choice([" The evidence is thin, so this is a weak prior rather than a prediction.",
                       " With so little precedent, do not lean on this heavily."])


def narrative(r: dict, g: int, rng: random.Random, explain: bool = False) -> str:
    """Prose about gate g for this target: cues, a cited precedent, a hedge, the censoring note.

    `explain` is for failure_attribution, which accounts for a stop that happened rather
    than a forecast, so its fallback must not claim a forecast exists.
    """
    cues = gate_cues(r, g)
    why = (rng.choice(LEADS["because"]).format(cues=oxford(cues[:2])) if cues
           else rng.choice(LEADS["nocue_explain" if explain else "nocue"]))
    return f"{why} {precedent_sentence(r, g, rng)}{hedge(r, rng)}{censoring_caveat(r)}"


def pipeline_forecast(r: dict, rng: random.Random, priors: dict[int, float]) -> dict | None:
    cond = r.get("gbm")
    if not cond or any(c is None or c != c for c in cond):
        return None
    q = rng.choice(PARAPHRASES["pipeline_forecast"])
    user = (f"{q}\n\nTarget features:\n{fmt_features(r)}\n\n"
            f"Archive evidence:\n{fmt_evidence(r)}"
            + (f"\n\nPrecedents:\n{fmt_precedents(r['precedents'])}" if r.get("precedents") else "")
            + f"\n\n{fmt_forecast(cond)}")
    g = forecast_drop_gate(cond)
    act = f"{LADDER[g]} -> {LADDER[g+1]} ({GATE_DESC[g]})"
    assistant = (f"Weakest point: {act}.\n"
                 + rng.choice(LEADS["weak"]).format(act=GATE_DESC[g]) + " "
                 + narrative(r, g, rng))
    return {"task": "pipeline_forecast", "target_id": r.get("target_id"), "gate": g, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}]}


def orthologue_ranking(group: list[dict], rng: random.Random) -> dict | None:
    if len(group) < 3:
        return None
    shown = group[:6]
    # censored records carry no information about tractability, so they sort last on
    # evidence rather than on the stage they happen to have reached
    order = sorted(shown, key=lambda p: (bool(p.get("censored")), -(p["max_stage"] or 0)))
    q = rng.choice(PARAPHRASES["orthologue_ranking"])
    rows = fmt_precedents(shown)
    user = (f"{q}\n\nCandidates from the same 30% identity cluster:\n{rows}\n\n"
            "Rank them and say what separates the top from the bottom.")
    lines = []
    for i, p in enumerate(order, 1):
        org = clean_organism(p.get("organism"))
        lines.append(f"  {i}. {p['target_id']}" + (f" ({org})" if org else "")
                     + f" reached {LADDER[p['max_stage']]}")
    n_cens = sum(1 for p in shown if p.get("censored"))
    # Both verbs, not just the first: the earlier fix produced "1 of the 3 records shown IS
    # censored ... and CARRY no information", which is the same agreement error one clause
    # further along.
    verb, verb2 = ("is", "carries") if n_cens == 1 else ("are", "carry")
    caveat = (f"\n\n{n_cens} of the {len(shown)} records shown {verb} censored at centre closure "
              f"and {verb2} no information about tractability; they are ranked last on evidence, "
              "not on merit." if n_cens else "")

    # "it is the only one to reach X" was asserted unconditionally and was FALSE in 8,029 of
    # 18,000 targets (44.6%), because nothing counted ties. Teaching a model to state a
    # checkable uniqueness claim that is wrong half the time is worse than saying less.
    top = order[0]
    top_stage = top["max_stage"] or 0
    # Count what the READER can see. Excluding censored records from this tally while still
    # printing them in the ranked list left 32 targets asserting "the only one to reach
    # soluble" directly above two rows that both read "reached soluble". The claim has to
    # be true of the list as displayed, not of a subset the sentence never mentions.
    n_at_top = sum(1 for p in shown if (p["max_stage"] or 0) == top_stage)
    if n_at_top == 1:
        lead = (f"{top['target_id']} is the strongest candidate: it is the only one in this "
                f"set to reach {LADDER[top_stage]}.")
    else:
        lead = (f"{top['target_id']} ranks first of the {n_at_top} candidates that reached "
                f"{LADDER[top_stage]}, which is as far as any of them got.")
    assistant = "Ranking:\n" + "\n".join(lines) + f"\n\n{lead}" + caveat
    return {"task": "orthologue_ranking", "target_id": top["target_id"], "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}]}


def construct_recommend(r: dict, rng: random.Random) -> dict | None:
    # A record whose host and tag are BOTH NaN used to pass this gate, because NaN is
    # truthy, and then produced a target reading "Host: nan ... Tag: nan, cleaved with nan."
    # That is why construct_recommend was the worst-affected task at 86% of held-out cases.
    if not (present(r.get("host")) or present(r.get("tag"))):
        return None
    q = rng.choice(PARAPHRASES["construct_recommend"])
    user = (f"{q}\n\nTarget features:\n{fmt_features(r)}\n\nArchive evidence:\n{fmt_evidence(r)}"
            + (f"\n\nPrecedents:\n{fmt_precedents(r['precedents'])}" if r.get("precedents") else ""))
    parts = []
    dn, dc = r.get("disorder_nterm") or 0, r.get("disorder_cterm") or 0
    if dn > 0.5 or dc > 0.5:
        trim = []
        if dn > 0.5:
            trim.append("trim the disordered N-terminus")
        if dc > 0.5:
            trim.append("trim the disordered C-terminus")
        parts.append("Boundaries: " + " and ".join(trim) + ".")
    elif present(r.get("construct_start")) and present(r.get("construct_end")):
        parts.append(f"Boundaries: residues {r['construct_start']} to {r['construct_end']}, "
                     "which is what worked in this cluster.")
    elif (r.get("target_construct_type") or "").startswith("truncated"):
        parts.append("Boundaries: a truncation, as the precedent in this cluster used; "
                     "the archive does not record which residues.")
    else:
        parts.append("Boundaries: start with the full-length ORF; there is no precedent for a truncation here.")
    if present(r.get("host")):
        parts.append(f"Host: {r['host']}, following the precedent in this cluster.")
    if present(r.get("tag")):
        parts.append(f"Tag: {r['tag']}"
                     + (f", cleaved with {r['protease']}" if present(r.get("protease")) else "")
                     + ".")
    if (r.get("tm_helices") or 0) >= 3:
        parts.append("This is predicted polytopic, so expect detergent screening to dominate the effort.")
    # was build_reason(r, max_stage >= 4): the prose tone was set by whether THIS target got
    # through, which is the answer, and at inference there is no answer to read.
    cond = r.get("gbm")
    if cond and all(c == c for c in cond):
        # round 5: aim the construct at the step the GBM says is weakest, and cite precedent
        g = forecast_drop_gate(cond)
        tail = f"The step to design for is {GATE_DESC[g]}. {narrative(r, g, rng)}"
    else:
        tail = build_reason(r, evidence_is_positive(r)) + censoring_caveat(r)
    assistant = "\n".join(parts) + "\n" + tail
    return {"task": "construct_recommend", "target_id": r.get("target_id"), "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}]}


def failure_attribution(r: dict, rng: random.Random) -> dict | None:
    """Explain a stop that is GIVEN in the prompt (Marc's call, 2026-09-27).

    Round 04's completion announced "In this archive it stopped at X" although the prompt
    never said so, which taught the model to state an outcome it was not given. The stage
    is now in the prompt and the answer explains it.

    Censored targets are excluded. Their earlier gates are legitimately in the loss as
    'cleared', so they reach this function, and describing a file closed at centre
    shutdown as a scientific failure breaks rule 2 (censored is never a negative).
    """
    g = r.get("max_stage")
    if g is None or g > 3 or r.get("target_censored"):
        return None
    q = rng.choice(PARAPHRASES["failure_attribution"])
    user = (f"{q}\n\nIn the archive this target stopped at {LADDER[g]}: it did not get through "
            f"{GATE_DESC[g]}.\n\nTarget features:\n{fmt_features(r)}\n\n"
            f"Archive evidence:\n{fmt_evidence(r)}"
            + (f"\n\nPrecedents:\n{fmt_precedents(r['precedents'])}" if r.get("precedents") else ""))
    assistant = (rng.choice(LEADS["stopped"]).format(act=GATE_DESC[g]) + " "
                 + narrative(r, g, rng, explain=True))
    return {"task": "failure_attribution", "target_id": r.get("target_id"), "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}]}


def approx_tokens(rec: dict) -> int:
    return sum(len(m["content"]) for m in rec["messages"]) // 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-records", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=42)
    # Without this the output path was a module constant, so EVERY run overwrote the live
    # corpus, including a small smoke build. On 2026-09-16 that would have rewritten
    # data/sft/train.jsonl underneath a training run that was eight hours in. A builder
    # whose only mode is "replace the thing currently in use" is a foot-gun.
    ap.add_argument("--out", type=Path, default=None,
                    help="write elsewhere than data/sft (use for smoke builds while a run is live)")
    args = ap.parse_args()
    # Rebind before the mkdir below, so every later reference (the mkdir, both writers and
    # the closing print) uses the override. Adding the flag without this would have been
    # worse than not adding it: an option that is silently ignored while a smoke build
    # quietly overwrites the live corpus is precisely the failure this guards against.
    global OUT
    if args.out is not None:
        OUT = args.out
    cfg = yaml.safe_load(CFG.read_text())
    mix = cfg.get("sft", {}).get("mix", {
        "gate_judgement": 0.45, "pipeline_forecast": 0.20, "orthologue_ranking": 0.15,
        "construct_recommend": 0.12, "failure_attribution": 0.08})
    hard_oversample = cfg.get("sft", {}).get("hard_oversample", 3)
    easy_cap = cfg.get("sft", {}).get("easy_negative_cap", 0.25)

    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB), read_only=True)
    print("loading L1 with features and context ...")
    df = con.execute("""
        SELECT l.target_id, l.centre, l.gate, l.label, l.in_loss, l.max_stage,
               s.split_cluster, s.split_temporal, s.censored AS target_censored, f.organism, f.superkingdom, f.seq_len, f.pi, f.gravy,
               f.net_charge_ph7, f.cys_count, f.tm_helices, f.signal_peptide,
               f.disorder_frac, f.disorder_nterm, f.disorder_cterm, f.low_complexity_frac,
               f.protein_types, f.target_construct_type, f.host, f.tag, f.protease,
               f.construct_start, f.construct_end, s.cluster_id, f.cluster70,
               f.n_cluster_precedents, f.n_close_precedents, f.cluster_censored_frac,
               c.n_precedents, c.n_precedents_cleared, c.cluster_base_rate
        FROM labels_l1 l
        JOIN splits s USING (target_id)
        LEFT JOIN features f USING (target_id)
        LEFT JOIN features_context c ON c.target_id = l.target_id AND c.gate = l.gate
        WHERE l.in_loss
    """).df()
    print(f"  {len(df):,} L1 rows in the loss")
    assert (df.label != "censored").all(), "a censored row reached the SFT corpus"

    # --- negative mining rules 3 and 4 -------------------------------------------------
    hard_ids = set(con.execute("""
        SELECT DISTINCT chosen_id FROM labels_l3 WHERE hard
        UNION SELECT DISTINCT rejected_id FROM labels_l3 WHERE hard
    """).df().iloc[:, 0])
    df["hard"] = df.target_id.isin(hard_ids)
    df["length_decile"] = (df.seq_len.rank(pct=True) * 10).fillna(0).astype(int)
    df["stratum"] = (df.tm_helices.fillna(0).clip(0, 3).astype(int).astype(str) + "|"
                     + df.superkingdom.fillna("") + "|" + df.length_decile.astype(str))

    train = df[df.split_cluster == "train"]
    neg = train[train.label == "failed"]
    pos = train[train.label == "cleared"]
    easy_neg = neg[~neg.hard]
    hard_neg = neg[neg.hard]

    # Rules 3 and 4 interact, so the cap is applied to the FINAL negative mass, after the
    # hard stratum is oversampled. Capping the distinct rows instead would leave easy
    # negatives at 10% once oversampling ran, and taking a floor of 25% of all negatives
    # (an earlier mistake here) defeated the cap entirely and let them reach 44%.
    n_hard_final = len(hard_neg) * hard_oversample
    n_easy_allowed = min(len(easy_neg),
                         int(easy_cap / (1 - easy_cap) * n_hard_final) if n_hard_final else len(easy_neg))
    if len(easy_neg) and n_easy_allowed < len(easy_neg):
        # stratified on (TM helix count, kingdom, length decile) so the cap does not
        # quietly delete a whole class of easy target
        frac = n_easy_allowed / len(easy_neg)
        # groupby(...).sample keeps every column; .apply(lambda g: g.sample(...)) drops
        # the grouping column and the stratum count could not then be reported.
        easy_keep = easy_neg.groupby("stratum", observed=True).sample(frac=frac, random_state=args.seed)
    else:
        easy_keep = easy_neg
    final_neg = len(easy_keep) + n_hard_final
    print(f"  negatives: {len(neg):,} distinct ({len(hard_neg):,} hard, {len(easy_neg):,} easy)")
    print(f"  hard oversampled {hard_oversample}x -> {n_hard_final:,};  easy capped to "
          f"{len(easy_keep):,} = {100*len(easy_keep)/max(final_neg,1):.0f}% of the final negative mass "
          f"(cap {easy_cap:.0%}), over {easy_keep.stratum.nunique():,} strata")
    pool = [pos, easy_keep] + [hard_neg] * hard_oversample
    train_mix = __import__("pandas").concat(pool, ignore_index=True)
    print(f"  training pool {len(train_mix):,} rows")

    # --- precedent lookup ---------------------------------------------------------------
    print("precedent tables ...")
    # Censored targets ARE included here, flagged. Spec rule 1: censored is excluded from
    # the loss but PRESENT IN CONTEXT and flagged in output. Drawing precedents from
    # labels_l2, which drops censored targets, meant no prompt ever contained one, so the
    # model could never learn to see and discount them.
    prec = con.execute("""
        SELECT s.cluster_id, c.target_id, c.centre, f.organism, c.max_stage, c.censored
        FROM splits s JOIN censoring c USING (target_id)
        LEFT JOIN features f USING (target_id)
        WHERE s.split_cluster = 'train' AND s.cluster_id IS NOT NULL
          AND c.max_stage IS NOT NULL
    """).df()
    by_cluster: dict[int, list[dict]] = {}
    for rec in prec.to_dict("records"):
        by_cluster.setdefault(rec["cluster_id"], []).append(rec)

    gate_rates = {}
    for tid, gate, rate in con.execute(
            "SELECT target_id, gate, cluster_base_rate FROM features_context").fetchall():
        if rate is not None:
            gate_rates.setdefault(tid, {})[gate] = rate

    # --- the GBM's forecast, which every round 5 prompt carries ---------------------------
    # Out-of-fold for training targets (scripts/06b_gbm_forecasts.py), so a prompt never
    # shows the optimistic numbers of a booster scoring a target it trained on.
    fc_path = PQ / "gbm_forecast.parquet"
    if not fc_path.exists():
        raise SystemExit(f"{fc_path} is missing: run scripts/06b_gbm_forecasts.py first")
    print("GBM forecasts ...")
    gbm_cluster: dict[str, list] = {}
    gbm_temporal: dict[str, list] = {}
    for tid, gate, pc, pt in con.execute(
            f"SELECT target_id, gate, p_cluster, p_temporal FROM read_parquet('{fc_path}')").fetchall():
        gbm_cluster.setdefault(tid, [None] * 8)[gate] = pc
        if pt is not None and pt == pt:
            gbm_temporal.setdefault(tid, [None] * 8)[gate] = pt
    gbm = gbm_cluster

    # --- generate -----------------------------------------------------------------------
    priors = {int(g): float(p) for g, p in con.execute("""
        SELECT l.gate, avg(CASE WHEN l.label = 'cleared' THEN 1.0 ELSE 0.0 END)
        FROM labels_l1 l JOIN splits s USING (target_id)
        WHERE l.in_loss AND s.split_cluster = 'train' GROUP BY 1
    """).fetchall()}
    print("  global gate priors: " + "  ".join(f"{g}:{p:.2f}" for g, p in sorted(priors.items())))

    print("generating records ...")
    records: list[dict] = []
    rows = train_mix.to_dict("records")
    rng.shuffle(rows)
    quota = {k: int(args.max_records * v) for k, v in mix.items()}
    made = dict.fromkeys(mix, 0)
    seen_forecast, seen_rank = set(), set()

    for r in rows:
        if all(made[k] >= quota[k] for k in mix):
            break
        cid = r.get("cluster_id")
        r["precedents"] = [p for p in by_cluster.get(cid, []) if p["target_id"] != r["target_id"]][:8] if cid else []
        r["gate_rates"] = gate_rates.get(r["target_id"], {})
        r["gbm"] = gbm.get(r["target_id"])
        if made["gate_judgement"] < quota["gate_judgement"]:
            rec = gate_judgement(r, rng, priors)
            if rec:
                records.append(rec); made["gate_judgement"] += 1
        if made["pipeline_forecast"] < quota["pipeline_forecast"] and r["target_id"] not in seen_forecast:
            seen_forecast.add(r["target_id"])
            rec = pipeline_forecast(r, rng, priors)
            if rec:
                records.append(rec); made["pipeline_forecast"] += 1
        if made["construct_recommend"] < quota["construct_recommend"]:
            rec = construct_recommend(r, rng)
            if rec:
                records.append(rec); made["construct_recommend"] += 1
        if made["failure_attribution"] < quota["failure_attribution"]:
            rec = failure_attribution(r, rng)
            if rec:
                records.append(rec); made["failure_attribution"] += 1
        if made["orthologue_ranking"] < quota["orthologue_ranking"] and cid and cid not in seen_rank:
            group = by_cluster.get(cid, [])
            if len(group) >= 3:
                seen_rank.add(cid)
                rec = orthologue_ranking(group, rng)
                if rec:
                    records.append(rec); made["orthologue_ranking"] += 1

    print("  " + "  ".join(f"{k}={made[k]:,}" for k in mix))

    # --- split and write ----------------------------------------------------------------
    # Validation and test come from the held-out CLUSTERS, never a random slice of these.
    val_rows = df[df.split_cluster == "valid"].to_dict("records")
    test_rows = df[df.split_cluster == "test"].to_dict("records")
    rng.shuffle(val_rows); rng.shuffle(test_rows)

    def eval_records(src, n):
        """The held-out sets span every task type, in the same mix as training.

        Built from gate_judgement alone, the generative eval's "cited a real precedent"
        check scored 0/40 by construction, because a gate judgement never names one.
        """
        out: list[dict] = []
        seen_rank_eval: set = set()
        quota_eval = {k: max(1, int(n * v)) for k, v in mix.items()}
        made_eval = dict.fromkeys(mix, 0)
        for r in src:
            if len(out) >= n:
                break
            cid = r.get("cluster_id")
            r["precedents"] = [p for p in by_cluster.get(cid, []) if p["target_id"] != r["target_id"]][:8] if cid else []
            r["gate_rates"] = gate_rates.get(r["target_id"], {})
            r["gbm"] = gbm.get(r["target_id"])
            for name, fn in (("gate_judgement", lambda: gate_judgement(r, rng, priors)),
                             ("pipeline_forecast", lambda: pipeline_forecast(r, rng, priors)),
                             ("construct_recommend", lambda: construct_recommend(r, rng)),
                             ("failure_attribution", lambda: failure_attribution(r, rng))):
                if made_eval[name] < quota_eval[name] and len(out) < n:
                    rec = fn()
                    if rec:
                        out.append(rec); made_eval[name] += 1
            if (made_eval["orthologue_ranking"] < quota_eval["orthologue_ranking"]
                    and cid and cid not in seen_rank_eval and len(out) < n):
                group = by_cluster.get(cid, [])
                if len(group) >= 3:
                    seen_rank_eval.add(cid)
                    rec = orthologue_ranking(group, rng)
                    if rec:
                        out.append(rec); made_eval["orthologue_ranking"] += 1
        return out

    valid = eval_records(val_rows, 2000)
    test = eval_records(test_rows, 2000)

    # A cluster-held-out test target has NO precedents by construction: whole clusters are
    # held out, so the training pool contains none of its relatives. That is correct for
    # measuring generalisation, but it means a cluster-test prompt never carries a
    # precedent table, and the generative eval's "cited a real precedent" and "flagged
    # censoring" checks then score 0/40 for a structural reason rather than a model one.
    #
    # The temporal split is where precedent is both present (82.3% of its test targets)
    # and honest: a 2014 target may have pre-2014 relatives. This second held-out set is
    # what eval_generative.py grades.
    temporal_prec: dict[int, list[dict]] = {}
    for rec in con.execute("""
        SELECT s.cluster_id, c.target_id, c.centre, f.organism, c.max_stage, c.censored
        FROM splits s JOIN censoring c USING (target_id)
        LEFT JOIN features f USING (target_id)
        WHERE s.split_temporal = 'train' AND s.cluster_id IS NOT NULL AND c.max_stage IS NOT NULL
    """).df().to_dict("records"):
        temporal_prec.setdefault(rec["cluster_id"], []).append(rec)

    temporal_rows = df[df.split_temporal == "test"].to_dict("records") if "split_temporal" in df else []
    rng.shuffle(temporal_rows)
    saved_by_cluster = by_cluster
    by_cluster = temporal_prec
    gbm = gbm_temporal   # pre-2014 boosters: the only honest forecast for a 2014+ target
    test_temporal = eval_records(temporal_rows, 2000)
    by_cluster = saved_by_cluster
    gbm = gbm_cluster
    with_prec = sum(1 for r in test_temporal if "Precedents:" in r["messages"][1]["content"])
    print(f"  temporal held-out: {len(test_temporal):,} records, "
          f"{with_prec:,} carrying a precedent table")

    for name, recs in (("train", records), ("valid", valid), ("test", test),
                       ("test_temporal", test_temporal)):
        # sort by length: MLX pads to the longest item in a batch
        recs.sort(key=approx_tokens)
        p = OUT / f"{name}.jsonl"
        with p.open("w") as fh:
            for rec in recs:
                fh.write(json.dumps({"messages": rec["messages"]}) + "\n")
        (OUT / f"{name}_tasks.jsonl").write_text("\n".join(json.dumps(
            {"task": r["task"], "tokens": approx_tokens(r),
             "target_id": r.get("target_id"), "gate": r.get("gate"),
             "label": r.get("label"), "p": r.get("p")}) for r in recs))
        toks = [approx_tokens(r) for r in recs]
        print(f"  {name:<6} {len(recs):>7,} records  tokens min {min(toks) if toks else 0} "
              f"median {sorted(toks)[len(toks)//2] if toks else 0} max {max(toks) if toks else 0}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
