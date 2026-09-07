#!/usr/bin/env python
"""07_build_sft.py: the supervised fine-tuning corpus.

Five task types in the specification's mix, chat JSONL, one {"messages": [...]} per line.

  gate_judgement       45%   will this target clear gate g, with a one-sentence reason
  pipeline_forecast    20%   the whole eight-gate vector plus the predicted bottleneck
  orthologue_ranking   15%   rank the given precedents by likely success, with reasoning
  construct_recommend  12%   boundaries, host and tag, justified from precedent
  failure_attribution   8%   which features drove the pessimism

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
        f"kingdom: {r['superkingdom']}" if r.get("superkingdom") else None,
        f"centre: {r['centre']}" if r.get("centre") else None,
        f"pI: {r['pi']:.2f}" if r.get("pi") is not None else None,
        f"GRAVY: {r['gravy']:.3f}" if r.get("gravy") is not None else None,
        f"net charge at pH 7: {r['net_charge_ph7']:+.1f}" if r.get("net_charge_ph7") is not None else None,
        f"cysteines: {r['cys_count']}" if r.get("cys_count") is not None else None,
        f"predicted TM helices: {r['tm_helices']}" if r.get("tm_helices") is not None else None,
        "signal peptide predicted" if r.get("signal_peptide") else None,
        f"disorder: {r['disorder_frac']:.0%} overall, {r['disorder_nterm']:.0%} N-terminal, "
        f"{r['disorder_cterm']:.0%} C-terminal" if r.get("disorder_frac") is not None else None,
        f"low-complexity: {r['low_complexity_frac']:.0%}" if r.get("low_complexity_frac") is not None else None,
        f"archive annotation: {r['protein_types']}" if r.get("protein_types") else None,
        f"construct type: {r['target_construct_type']}" if r.get("target_construct_type") else None,
        f"host: {r['host']}" if r.get("host") else None,
        f"tag: {r['tag']}" if r.get("tag") else None,
    ]
    return "\n".join(f"  {b}" for b in bits if b)


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
        line += f"\n  {cens:.0%} of this cluster's records are censored"
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


def gate_judgement(r: dict, rng: random.Random, priors: dict[int, float]) -> dict | None:
    g = r["gate"]
    base = r.get("cluster_base_rate")
    q = rng.choice(PARAPHRASES["gate_judgement"]).format(action=GATE_DESC[g])
    user = (f"{q}\n\nTarget features:\n{fmt_features(r)}\n\n"
            f"Gate under consideration: {LADDER[g]} -> {LADDER[g+1]} ({GATE_DESC[g]})\n\n"
            f"Archive evidence:\n{fmt_evidence(r)}"
            + (f"\n  historical clearance of this gate in this cluster: {base:.0%}"
               if base is not None else "")
            + (f"\n\nPrecedents:\n{fmt_precedents(r['precedents'])}" if r.get("precedents") else ""))
    p = calibrated_p(r, priors)
    reason = build_reason(r, p >= 0.5)
    assistant = (f"Probability of getting from {LADDER[g]} to {LADDER[g+1]}: {p:.2f}\n"
                 f"{reason}{censoring_caveat(r)}")
    return {"task": "gate_judgement", "target_id": r["target_id"], "gate": g,
            "label": r.get("label"), "p": round(p, 4), "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}]}


def censoring_caveat(r: dict) -> str:
    """Spec rule 1: censored is excluded from the loss, present in context, and FLAGGED IN
    OUTPUT. Only the ranking task said so, and the generative eval scored 0 of 19 cases
    that had a censored precedent in context."""
    cens = [p for p in (r.get("precedents") or []) if p.get("censored")]
    frac = r.get("cluster_censored_frac") or 0
    if cens:
        n, tot = len(cens), len(r.get("precedents") or [])
        return (f" Note that {n} of {tot} precedents here are censored: their files closed "
                "when the centre stopped reporting, so they are evidence of nothing either "
                "way and the real base is thinner than the count suggests.")
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
        tail = f" despite {', '.join(cues[:2])}" if cues else ""
        return f"{head}{tail}."
    head = "The precedent in this cluster is discouraging" if n >= 5 else \
           "There is little precedent here, and the intrinsic properties are unhelpful"
    tail = f", and this target carries {', '.join(cues[:3])}" if cues else ""
    return f"{head}{tail}."


def pipeline_forecast(r: dict, rng: random.Random) -> dict:
    reached = r["max_stage"]
    rates = r.get("gate_rates") or {}
    q = rng.choice(PARAPHRASES["pipeline_forecast"])
    user = (f"{q}\n\nTarget features:\n{fmt_features(r)}\n\n"
            f"Archive evidence:\n{fmt_evidence(r)}"
            + (f"\n\nPrecedents:\n{fmt_precedents(r['precedents'])}" if r.get("precedents") else ""))
    lines, surv = [], 1.0
    bottleneck, worst = 0, 1.1
    for g in range(8):
        cond = rates.get(g)
        if cond is None:
            cond = 0.85 if g < reached else 0.35
        cond = min(0.97, max(0.05, cond))
        surv *= cond
        lines.append(f"  {LADDER[g]} -> {LADDER[g+1]}: {cond:.2f} conditional, {surv:.2f} cumulative")
        if cond < worst:
            worst, bottleneck = cond, g
    assistant = ("Gate-by-gate outlook:\n" + "\n".join(lines) +
                 f"\n\nPredicted bottleneck: {LADDER[bottleneck]} -> {LADDER[bottleneck+1]}"
                 f" ({GATE_DESC[bottleneck]}).\n{build_reason(r, reached >= 5)}"
                 f"{censoring_caveat(r)}")
    return {"task": "pipeline_forecast", "messages": [
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
    caveat = (f"\n\n{n_cens} of these records are censored at centre closure and carry no "
              "information about tractability; they are ranked last on evidence, not on merit."
              if n_cens else "")
    assistant = ("Ranking:\n" + "\n".join(lines) +
                 f"\n\n{order[0]['target_id']} is the strongest candidate: it is the only one "
                 f"in this set to reach {LADDER[order[0]['max_stage']]}." + caveat)
    return {"task": "orthologue_ranking", "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}]}


def construct_recommend(r: dict, rng: random.Random) -> dict | None:
    if not (r.get("host") or r.get("tag")):
        return None
    q = rng.choice(PARAPHRASES["construct_recommend"])
    user = (f"{q}\n\nTarget features:\n{fmt_features(r)}\n\nArchive evidence:\n{fmt_evidence(r)}")
    parts = []
    dn, dc = r.get("disorder_nterm") or 0, r.get("disorder_cterm") or 0
    if dn > 0.5 or dc > 0.5:
        trim = []
        if dn > 0.5:
            trim.append("trim the disordered N-terminus")
        if dc > 0.5:
            trim.append("trim the disordered C-terminus")
        parts.append("Boundaries: " + " and ".join(trim) + ".")
    elif r.get("construct_start") and r.get("construct_end"):
        parts.append(f"Boundaries: residues {r['construct_start']} to {r['construct_end']}, "
                     "which is what worked in this cluster.")
    elif (r.get("target_construct_type") or "").startswith("truncated"):
        parts.append("Boundaries: a truncation, as the precedent in this cluster used; "
                     "the archive does not record which residues.")
    else:
        parts.append("Boundaries: start with the full-length ORF; there is no precedent for a truncation here.")
    if r.get("host"):
        parts.append(f"Host: {r['host']}, following the precedent in this cluster.")
    if r.get("tag"):
        parts.append(f"Tag: {r['tag']}" + (f", cleaved with {r['protease']}" if r.get("protease") else "") + ".")
    if (r.get("tm_helices") or 0) >= 3:
        parts.append("This is predicted polytopic, so expect detergent screening to dominate the effort.")
    assistant = ("\n".join(parts) + "\n" + build_reason(r, (r.get("max_stage") or 0) >= 4)
                 + censoring_caveat(r))
    return {"task": "construct_recommend", "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}]}


def failure_attribution(r: dict, rng: random.Random) -> dict | None:
    if (r.get("max_stage") or 0) > 3:
        return None
    q = rng.choice(PARAPHRASES["failure_attribution"])
    user = (f"{q}\n\nTarget features:\n{fmt_features(r)}\n\nArchive evidence:\n{fmt_evidence(r)}")
    assistant = (build_reason(r, False) + " In this archive it stopped at "
                 f"{LADDER[r['max_stage']]}, which is consistent with that reading."
                 + censoring_caveat(r))
    return {"task": "failure_attribution", "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}]}


def approx_tokens(rec: dict) -> int:
    return sum(len(m["content"]) for m in rec["messages"]) // 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-records", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
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
               s.split_cluster, s.split_temporal, f.organism, f.superkingdom, f.seq_len, f.pi, f.gravy,
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
        if made["gate_judgement"] < quota["gate_judgement"]:
            rec = gate_judgement(r, rng, priors)
            if rec:
                records.append(rec); made["gate_judgement"] += 1
        if made["pipeline_forecast"] < quota["pipeline_forecast"] and r["target_id"] not in seen_forecast:
            seen_forecast.add(r["target_id"])
            records.append(pipeline_forecast(r, rng)); made["pipeline_forecast"] += 1
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
            for name, fn in (("gate_judgement", lambda: gate_judgement(r, rng, priors)),
                             ("pipeline_forecast", lambda: pipeline_forecast(r, rng)),
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
    test_temporal = eval_records(temporal_rows, 2000)
    by_cluster = saved_by_cluster
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
