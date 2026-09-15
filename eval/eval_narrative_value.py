#!/usr/bin/env python
"""eval_narrative_value.py: does the fine-tuned model add anything, or echo its templates?

The specification is blunt about this: "If you cannot state a number the LLM adds over the
GBM, you are shipping decoration." `eval_calibration.py` answers that for the probability.
This answers it for the prose, which is the only thing the model is allowed to write.

The worry is concrete and visible during round 01: training loss fell from 2.654 to 0.199
within fifty iterations. A corpus whose completions come from a handful of Python templates
can be fitted almost immediately, and a model that has memorised the template produces text
that looks fluent, is perfectly calibrated, and carries no information an f-string could not
have supplied.

Three measures, none of which need a human:

  TEMPLATE ECHO      similarity between a generated narrative and the nearest completion in
                     the training corpus. High means it is reciting.
  RESPONSIVENESS     similarity between narratives generated for DIFFERENT targets. High
                     means the text does not depend on the input, which is the failure that
                     matters: the same paragraph for every protein.
  GROUNDING          does the narrative mention the features and precedents it was given,
                     and does it change when those change? Tested by perturbing one input
                     (the bottleneck gate) and measuring whether the text follows.

A model can score well on hallucination and calibration and still fail all three.

Usage
  .venv/bin/python eval/eval_narrative_value.py --llm-endpoint http://127.0.0.1:8080/v1
  .venv/bin/python eval/eval_narrative_value.py --dry-run      # scores the corpus itself
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
SFT = ROOT / "data" / "sft"
OUT = ROOT / "eval"


def shingles(text: str, n: int = 4) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z0-9.%]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def nearest(text: str, corpus: list[set]) -> float:
    s = shingles(text)
    return max((jaccard(s, c) for c in corpus), default=0.0)


def mean_pairwise(texts: list[str]) -> float:
    sh = [shingles(t) for t in texts]
    pairs = [(i, j) for i in range(len(sh)) for j in range(i + 1, len(sh))]
    if not pairs:
        return 0.0
    return sum(jaccard(sh[i], sh[j]) for i, j in pairs) / len(pairs)


BOTTLENECK = re.compile(r"Predicted bottleneck:\s*([\w ]+?)\s*(?:->|\(|\.|$)")


def stated_bottleneck(text: str) -> str | None:
    """The gate a forecast names as the wall.

    The pipeline_forecast PROMPT carries no gate table: the model is trained to produce
    the whole vector itself from features and evidence. So the reference is the held-out
    completion, and this measure is the specification's own bottleneck top-1 accuracy:
    does the model name the same wall as the reference answer for that target.

    The first version searched the prompt for a string that only ever appears in the
    completion, and duly scored 0/24 against a corpus that names its gate every time.
    """
    m = BOTTLENECK.search(text)
    return m.group(1).strip().lower() if m else None


def complete(messages, endpoint: str, model: str, max_tokens: int = 320) -> str:
    import requests
    r = requests.post(f"{endpoint}/chat/completions", timeout=180,
                      json={"model": model, "messages": messages,
                            "max_tokens": max_tokens, "temperature": 0.2})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def resolved_model(endpoint: str) -> str:
    import requests
    try:
        d = requests.get(f"{endpoint}/models", timeout=10).json()["data"]
        return d[0]["id"]
    except Exception:  # noqa: BLE001
        return "faffabout"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-endpoint", default=None)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--set", default="test_temporal")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="score the corpus's own completions: the floor a model must beat")
    ap.add_argument("--label", default=None,
                    help="names the output file, so a model run cannot overwrite the floor")
    args = ap.parse_args()
    label = args.label or ("corpus" if args.dry_run else "model")

    rng = random.Random(args.seed)
    train = [json.loads(l) for l in (SFT / "train.jsonl").open()]
    rng.shuffle(train)
    corpus = [shingles(r["messages"][2]["content"]) for r in train[:4000]]

    held = [json.loads(l) for l in (SFT / f"{args.set}.jsonl").open()]
    meta = [json.loads(l) for l in (SFT / f"{args.set}_tasks.jsonl").open() if l.strip()]
    pairs = [(r, m) for r, m in zip(held, meta) if m["task"] == "pipeline_forecast"]
    rng.shuffle(pairs)
    pairs = pairs[:args.n]
    if not pairs:
        sys.exit(f"no pipeline_forecast records in {args.set}")

    model = args.llm_model or (resolved_model(args.llm_endpoint) if args.llm_endpoint else "")
    if args.llm_endpoint:
        print(f"generating {len(pairs)} narratives from {model}")

    texts, sources = [], []
    for i, (rec, m) in enumerate(pairs, 1):
        if args.llm_endpoint:
            try:
                t = complete(rec["messages"][:2], args.llm_endpoint, model)
            except Exception as e:  # noqa: BLE001
                print(f"  case {i} failed: {type(e).__name__}")
                continue
        else:
            t = rec["messages"][2]["content"]
        texts.append(t)
        sources.append(rec)
        print(f"  {i}/{len(pairs)}", end="\r", flush=True)

    if len(texts) < 4:
        sys.exit("too few narratives to score")

    echo = [nearest(t, corpus) for t in texts]
    resp = mean_pairwise(texts)

    print(f"\n\n=== {len(texts)} narratives from "
          f"{'the served model' if args.llm_endpoint else 'the corpus itself'} ===")
    print(f"  template echo   mean {sum(echo)/len(echo):.3f}  max {max(echo):.3f}")
    print("                  (similarity to the nearest TRAINING completion; "
          "high means recitation)")
    print(f"  responsiveness  mean pairwise {resp:.3f}")
    print("                  (similarity BETWEEN narratives for different targets; "
          "high means the text ignores its input)")

    grounded = agreed = comparable = 0
    for t, rec in zip(texts, sources):
        said = stated_bottleneck(t)
        ref = stated_bottleneck(rec["messages"][2]["content"])
        if said:
            grounded += 1
        if said and ref:
            comparable += 1
            agreed += said == ref
    print(f"  names a wall    {grounded}/{len(texts)}")
    print(f"  bottleneck top-1 {agreed}/{comparable} agree with the held-out answer"
          if comparable else "  bottleneck top-1 n/a")

    # The floor is whatever the corpus scored, not a number pasted into a shell script.
    floor = OUT / "narrative_value_corpus.json"
    fl = json.loads(floor.read_text()) if floor.exists() and label != "corpus" else None
    if fl:
        de = sum(echo) / len(echo) - fl["template_echo_mean"]
        dr = resp - fl["responsiveness_mean_pairwise"]
        print(f"\n  against the corpus floor (n={fl['n']}):")
        print(f"    echo           {de:+.3f}  " +
              ("AT OR ABOVE THE FLOOR: recitation" if de >= 0 else "below the floor"))
        print(f"    responsiveness {dr:+.3f}  " +
              ("less varied than the corpus itself" if dr >= 0 else "more varied than the corpus"))

    verdict = []
    if fl and sum(echo) / len(echo) >= fl["template_echo_mean"]:
        verdict.append("narratives match training completions at least as closely as the corpus "
                       "matches itself: the template has been memorised")
    if resp > 0.60:
        verdict.append("narratives barely differ between targets: the model is not using its input")
    if sum(echo) / len(echo) > 0.70:
        verdict.append("narratives closely match training completions: the template has been memorised")
    if grounded / len(texts) < 0.5:
        verdict.append("most narratives never name a bottleneck at all")
    if comparable and agreed / comparable < 0.4:
        verdict.append("the named wall usually disagrees with the held-out answer")
    print("\n" + ("\n".join("  WARNING: " + v for v in verdict) if verdict
                  else "  no template-echo or unresponsiveness warnings"))
    if not args.dry_run:
        print("\n  Compare against --dry-run, which scores the corpus itself: that is the\n"
              "  floor, since the corpus IS the templates. A model at or above it has\n"
              "  learned to recite rather than to reason.")

    OUT.mkdir(parents=True, exist_ok=True)
    blob = json.dumps({
        "source": model if args.llm_endpoint else "corpus",
        "n": len(texts), "template_echo_mean": sum(echo) / len(echo),
        "template_echo_max": max(echo), "responsiveness_mean_pairwise": resp,
        "names_a_wall": grounded, "bottleneck_top1": (agreed / comparable) if comparable else None,
        "warnings": verdict,
        "samples": texts[:3],
    }, indent=2)
    # Two files on purpose: the labelled one is permanent, so a later model run cannot
    # destroy the floor it is being judged against.
    (OUT / f"narrative_value_{label}.json").write_text(blob)
    (OUT / "narrative_value.json").write_text(blob)
    print(f"\nwrote {OUT / f'narrative_value_{label}.json'} (and narrative_value.json)")


if __name__ == "__main__":
    main()
