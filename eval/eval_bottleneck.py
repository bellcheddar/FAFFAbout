"""Bottleneck top-1 against where targets ACTUALLY stopped: GBM vs LLM vs the corpus template.

`eval_narrative_value.py` scores "bottleneck top-1" as agreement between the model's named
wall and the REFERENCE COMPLETION's, and the reference is written by a template from shrunk
cluster base rates. So its 15/24 for round 04 measures imitation of that formula, not
whether the named wall is where the protein died. This script scores every contender
against the archive's own answer: for an uncensored target that never deposited, the wall
is the gate out of its `max_stage` (gate g is the transition from stage g to g+1).

Contenders, all on the same targets:
  majority   always the most common wall among TRAINING-split targets (the floor)
  template   the corpus's reference completion (lowest conditional)
  gbm_min    temporal-split GBM, gate with the lowest conditional probability
  gbm_drop   temporal-split GBM, gate losing the most survival mass (app/predict.py's rule)
  llm        the served adapter's own forecast, if --llm-endpoint is given

The saved boosters in baseline/models were trained on the CLUSTER split, whose training
rows include these temporal-test targets, so scoring them here would leak. The temporal
boosters are trained once and saved under the `temporal_` prefix, beside the app's.

Usage
  .venv/bin/python eval/eval_bottleneck.py                                  # no LLM
  .venv/bin/python eval/eval_bottleneck.py --llm-endpoint http://127.0.0.1:8080/v1 --label round04
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))
import gbm_baseline as gb  # noqa: E402

SFT = ROOT / "data" / "sft"
OUT = ROOT / "eval"
LADDER = gb.LADDER
BOTTLENECK = re.compile(r"(?:Predicted bottleneck|Weakest point):\s*([a-z]+)\s*->")
PREFIX = "temporal_"


def named_gate(text: str) -> int | None:
    m = BOTTLENECK.search(text or "")
    return LADDER.index(m.group(1)) if m and m.group(1) in LADDER[:-1] else None


def temporal_boosters(con) -> dict:
    import lightgbm as lgb
    if not (gb.MODELS / f"{PREFIX}schema.json").exists():
        print("training temporal-split boosters (once) ...")
        gb.run(con, "split_temporal", "test", "temporal", gb.PREDICTION_TIME, save=True, prefix=PREFIX)
    return {g: lgb.Booster(model_file=str(gb.MODELS / f"{PREFIX}gate_{g}.txt"))
            for g in range(8) if (gb.MODELS / f"{PREFIX}gate_{g}.txt").exists()}


def gbm_conditionals(con, ids: list[str], bst: dict) -> dict[str, list[float]]:
    """Per-gate conditional P(clear) for every target at all eight gates."""
    schema = json.loads((gb.MODELS / f"{PREFIX}schema.json").read_text())
    con.execute("CREATE OR REPLACE TEMP TABLE ids AS SELECT unnest(?) AS target_id", [ids])
    df = con.execute("""
        SELECT i.target_id, c.gate, f.* EXCLUDE (target_id, organism, seq_md5),
               c.n_precedents, c.n_precedents_cleared, c.cluster_base_rate
        FROM ids i JOIN features_context c USING (target_id)
        LEFT JOIN features f USING (target_id)
    """).df()
    X = gb.prep(df, schema["features"])
    for col, levels in schema["categorical"].items():
        if col in X:
            X[col] = pd.Categorical(df[col].astype("string"), categories=levels)
    X = X[schema["features"]]
    out = {t: [np.nan] * 8 for t in ids}
    for g, b in bst.items():
        m = (df.gate == g).values
        p = b.predict(X[m], num_iteration=b.best_iteration or None)
        for t, v in zip(df.target_id[m], p):
            out[t][g] = float(min(0.99, max(0.01, v)))
    return out


def by_drop(cond: list[float]) -> int:
    surv, run = [1.0], 1.0
    for c in cond:
        run *= c
        surv.append(run)
    return int(np.argmax([surv[i] - surv[i + 1] for i in range(8)]))


def llm_forecast(messages, endpoint: str, model: str) -> str:
    import requests
    body = {"model": model, "messages": messages, "max_tokens": 320, "temperature": 0.0}
    if os.environ.get("FAFFABOUT_LLM_ADAPTER"):  # the only way to attach it: see app/llm.py
        body["adapters"] = os.environ["FAFFABOUT_LLM_ADAPTER"]
    r = requests.post(f"{endpoint}/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-endpoint", default=None)
    ap.add_argument("--llm-model", default="default_model")
    ap.add_argument("--label", default="gbm_only")
    args = ap.parse_args()

    con = duckdb.connect(str(ROOT / "data" / "faffabout.duckdb"), read_only=True)
    held = [json.loads(l) for l in (SFT / "test_temporal.jsonl").open()]
    meta = [json.loads(l) for l in (SFT / "test_temporal_tasks.jsonl").open() if l.strip()]
    recs = {m["target_id"]: r for r, m in zip(held, meta) if m["task"] == "pipeline_forecast"}
    truth = dict(con.execute("""
        SELECT target_id, max_stage FROM splits
        WHERE target_id IN (SELECT unnest(?)) AND NOT censored AND max_stage < 8
    """, [list(recs)]).fetchall())
    ids = sorted(truth)
    majority = con.execute("""
        SELECT max_stage FROM splits WHERE split_temporal = 'train' AND NOT censored AND max_stage < 8
        GROUP BY 1 ORDER BY count(*) DESC LIMIT 1
    """).fetchone()[0]
    print(f"{len(ids)} uncensored, undeposited pipeline_forecast targets in test_temporal; "
          f"training-split majority wall = gate {majority} ({LADDER[majority]} -> {LADDER[majority+1]})")

    cond = gbm_conditionals(con, ids, temporal_boosters(con))
    picks = {"majority": {t: majority for t in ids},
             "template": {t: named_gate(recs[t]["messages"][2]["content"]) for t in ids},
             "gbm_min": {t: int(np.nanargmin(cond[t])) for t in ids},
             "gbm_drop": {t: by_drop(cond[t]) for t in ids}}
    if args.llm_endpoint:
        llm = {}
        for i, t in enumerate(ids, 1):
            try:
                llm[t] = named_gate(llm_forecast(recs[t]["messages"][:2], args.llm_endpoint, args.llm_model))
            except Exception as e:  # noqa: BLE001
                print(f"  {t} failed: {type(e).__name__}")
                llm[t] = None
            print(f"  llm {i}/{len(ids)}", end="\r", flush=True)
        picks["llm"] = llm
        print()

    res = {"n": len(ids), "truth_distribution": dict(sorted(Counter(truth.values()).items())), "scores": {}}
    print(f"\ntrue walls: " + ", ".join(f"g{g}:{n}" for g, n in sorted(Counter(truth.values()).items())))
    print(f"\n{'contender':10s} {'top-1':>7s} {'named':>7s}")
    for name, p in picks.items():
        named = [t for t in ids if p[t] is not None]
        hit = sum(p[t] == truth[t] for t in named)
        res["scores"][name] = {"top1": hit / len(ids), "hits": hit, "named": len(named),
                               "picked": dict(sorted(Counter(p[t] for t in named).items()))}
        print(f"{name:10s} {hit/len(ids):7.1%} {len(named):7d}   picks {dict(sorted(Counter(p[t] for t in named).items()))}")
    if "llm" in picks:
        both = [t for t in ids if picks["llm"][t] is not None and picks["template"][t] is not None]
        agree = sum(picks["llm"][t] == picks["template"][t] for t in both)
        res["llm_agrees_with_template"] = agree / len(both) if both else None
        print(f"\nllm agrees with the template's wall on {agree}/{len(both)} (the old 'bottleneck top-1')")
    (OUT / f"bottleneck_{args.label}.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / f'bottleneck_{args.label}.json'}")


if __name__ == "__main__":
    main()
