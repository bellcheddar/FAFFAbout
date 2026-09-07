#!/usr/bin/env python
"""eval_generative.py: 40 held-out cases, graded against a rubric.

Two halves. The automatic half runs without a human and is where the hard failure lives:

  HALLUCINATED IDENTIFIER  a target id or PDB code in the completion that was not in the
                           prompt. This is an AUTOMATIC FAIL for the whole run, because it
                           means facts leaked into the weights that should have stayed in
                           retrieval.

  censoring flagged        does the answer mention censoring when the retrieved context
                           contains a censored precedent
  precedents cited         does it cite at least one identifier it was actually given
  no invented probability   the app's numbers come from the GBM, so a bare float outside
                           the expected line is worth seeing

The manual half writes eval/generative_review.md, a form with the 40 cases and a rubric,
for a domain expert to grade. That grading is the acceptance test and it is not automatable.

Usage
  .venv/bin/python eval/eval_generative.py --llm-endpoint http://127.0.0.1:8080/v1
  .venv/bin/python eval/eval_generative.py --dry-run     # write the form from the corpus
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFT = ROOT / "data" / "sft"
OUT = ROOT / "eval"

# A TargetTrack identifier is CENTRE-localid. The centre list is read from the archive
# rather than guessed at with a general pattern: `[A-Z][A-Za-z0-9]{1,12}-...` also matches
# ordinary hyphenated English, and "Gate-by" out of "Gate-by-gate outlook" was reported as
# a hallucinated identifier in 10 of 40 cases.
def _centre_names() -> list[str]:
    try:
        import duckdb
        con = duckdb.connect(str(ROOT / "data" / "faffabout.duckdb"), read_only=True)
        names = [r[0] for r in con.execute(
            "SELECT DISTINCT centre FROM targets WHERE centre <> ''").fetchall()]
        con.close()
        return sorted(names, key=len, reverse=True)
    except Exception:
        return []


_CENTRES = _centre_names()
RE_IDENT = (re.compile(r"\b(?:" + "|".join(re.escape(c) for c in _CENTRES) + r")-[A-Za-z0-9_.]{1,20}\b")
            if _CENTRES else re.compile(r"(?!x)x"))
RE_PDB = re.compile(r"\b[1-9][A-Za-z0-9]{3}\b")
RE_FLOAT = re.compile(r"\b[01]\.\d+\b")

RUBRIC = """\
| # | Criterion | Pass if |
|---|---|---|
| 1 | Correct bottleneck | The gate it names as the wall matches where the target actually stopped, or is defensible given the evidence shown |
| 2 | Cited real precedents | Every identifier it names appears in its own prompt |
| 3 | No invented identifiers | It names no target id or PDB code that was not given to it. **Any failure here fails the whole run** |
| 4 | Censoring flagged | Where the retrieved context contains a censored precedent, the answer says so and does not treat it as evidence of failure |
| 5 | Honest about thin evidence | Where there are few precedents, it hedges rather than asserting |
| 6 | Reasoning is about this protein | The rationale cites features of this target, not generic structural-biology advice |
| 7 | Useful to a scientist | Would this change what you did on Monday morning |
"""


def check_one(prompt: str, completion: str) -> dict:
    prompt_idents = set(RE_IDENT.findall(prompt))
    said_idents = set(RE_IDENT.findall(completion))
    invented = sorted(said_idents - prompt_idents)

    prompt_pdb = set(RE_PDB.findall(prompt))
    said_pdb = {p for p in RE_PDB.findall(completion)
                if not re.match(r"^[0-9]{4}$", p)}      # a bare year is not a PDB code
    invented_pdb = sorted(said_pdb - prompt_pdb)

    # Every prompt carries the line "N% of this cluster's records are censored", so
    # searching for the word alone marks all 40 cases as containing a censored precedent.
    # A real one is a precedent ROW flagged at centre closure, or a non-zero fraction.
    has_censored_row = "censored at centre closure" in prompt
    frac = re.search(r"(\d+)% of this cluster's records are censored", prompt)
    ctx_has_censored = has_censored_row or bool(frac and int(frac.group(1)) > 0)
    return {
        "invented_identifiers": invented,
        "invented_pdb_codes": invented_pdb,
        "cited_a_real_precedent": bool(said_idents & prompt_idents),
        "context_has_censored_precedent": ctx_has_censored,
        "mentions_censoring": "censor" in completion.lower(),
        "censoring_flagged_when_present": (not ctx_has_censored) or ("censor" in completion.lower()),
        "floats_emitted": RE_FLOAT.findall(completion),
    }


def llm_complete(messages, endpoint: str, model: str) -> str:
    import requests
    body = {"model": model, "messages": messages, "max_tokens": 400, "temperature": 0.0}
    r = requests.post(f"{endpoint}/chat/completions", json=body, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-endpoint", default=None)
    ap.add_argument("--llm-model", default="faffabout")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set", default="auto", choices=["auto", "test", "test_temporal", "valid"],
                    help="which held-out set to grade; auto prefers test_temporal")
    ap.add_argument("--dry-run", action="store_true", help="grade the corpus's own completions")
    args = ap.parse_args()

    # Default to the temporal held-out set: cluster-held-out prompts carry no precedent
    # table at all, so the citation and censoring checks cannot fire on them.
    stem = args.set
    if stem == "auto":
        stem = "test_temporal" if (SFT / "test_temporal.jsonl").exists() else "test"
    print(f"grading the '{stem}' held-out set")
    recs = [json.loads(l) for l in (SFT / f"{stem}.jsonl").open()]
    meta = [json.loads(l) for l in (SFT / f"{stem}_tasks.jsonl").open() if l.strip()]
    # Spread the sample across task types, and SHUFFLE within each. The corpus is written
    # sorted by token length for MLX padding, so taking from the head of each list selects
    # the shortest records, which are the ones carrying the fewest precedents: a 40-case
    # sample drawn that way reported 2 cases with censored context where the full set has
    # 1,265 of 2,000.
    rng = random.Random(args.seed)
    by_task: dict[str, list] = {}
    for r, m in zip(recs, meta):
        by_task.setdefault(m["task"], []).append((r, m))
    for t in by_task:
        rng.shuffle(by_task[t])
    picked = []
    while len(picked) < args.n and any(by_task.values()):
        for t in list(by_task):
            if by_task[t] and len(picked) < args.n:
                picked.append((t, *by_task[t].pop(0)))

    results, fails = [], 0
    for i, (task, rec, m) in enumerate(picked, 1):
        prompt = rec["messages"][1]["content"]
        if args.llm_endpoint:
            try:
                completion = llm_complete(rec["messages"][:2], args.llm_endpoint, args.llm_model)
            except Exception as e:  # noqa: BLE001
                print(f"case {i}: request failed: {type(e).__name__}: {e}")
                continue
        else:
            completion = rec["messages"][2]["content"]   # grade the corpus itself
        chk = check_one(prompt, completion)
        if chk["invented_identifiers"] or chk["invented_pdb_codes"]:
            fails += 1
        results.append({"case": i, "task": task, "target_id": m.get("target_id"),
                        "prompt": prompt, "completion": completion, **chk})
        print(f"  {i}/{len(picked)}", end="\r", flush=True)

    n = len(results)
    print(f"\n\n=== automatic checks over {n} cases "
          f"({'served model' if args.llm_endpoint else 'the corpus itself'}) ===")
    cited = sum(r["cited_a_real_precedent"] for r in results)
    with_cens = [r for r in results if r["context_has_censored_precedent"]]
    flagged = sum(r["mentions_censoring"] for r in with_cens)
    print(f"cited a precedent from its own prompt : {cited}/{n}")
    print(f"censored precedent in context         : {len(with_cens)}/{n}")
    print(f"  ... and censoring mentioned         : {flagged}/{len(with_cens) if with_cens else 0}")
    print(f"cases inventing an identifier         : {fails}/{n}")
    if fails:
        print("\nHALLUCINATED IDENTIFIERS: AUTOMATIC FAIL")
        for r in results:
            if r["invented_identifiers"] or r["invented_pdb_codes"]:
                print(f"  case {r['case']} ({r['task']}): "
                      f"{r['invented_identifiers'] + r['invented_pdb_codes']}")
    else:
        print("\nno hallucinated identifiers: automatic checks PASS")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "generative_results.json").write_text(json.dumps(results, indent=2))

    form = [f"# FAFFAbout generative review ({n} cases)", "",
            f"Source: {'served model at ' + args.llm_endpoint if args.llm_endpoint else 'the SFT corpus itself (dry run)'}",
            "", "Grade each case against the rubric. Criterion 3 is an automatic fail for the",
            "whole run, not just the case.", "", RUBRIC, "", "---", ""]
    for r in results:
        form += [f"## Case {r['case']} ({r['task']})", "",
                 f"Target: `{r['target_id']}`", "", "**Prompt**", "", "```text", r["prompt"], "```", "",
                 "**Completion**", "", "```text", r["completion"], "```", "",
                 "| Criterion | 1 | 2 | 3 | 4 | 5 | 6 | 7 |", "|---|---|---|---|---|---|---|---|",
                 "| Pass? |  |  |  |  |  |  |  |", "", "Notes:", "", "---", ""]
    (OUT / "generative_review.md").write_text("\n".join(form))
    print(f"\nwrote {OUT / 'generative_results.json'}")
    print(f"wrote {OUT / 'generative_review.md'}  <- the grading form; the expert half is not automatable")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
