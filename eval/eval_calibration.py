#!/usr/bin/env python
"""eval_calibration.py: is the probability honest, and what does the LLM add over the GBM?

Reports on every split:

  Brier score              the headline: is the probability honest
  Expected calibration err  10-bin, the number to quote
  AUROC per gate            comparability with the GBM
  Bottleneck top-1          does it name the right wall
  GBM delta                 the number that justifies the fine-tune

Runs with no model at all (GBM only), or against a served adapter with --llm-endpoint.
If no number can be stated for what the language model adds over the gradient-booster,
the language model is decoration and this script is where that becomes visible.

Usage
  .venv/bin/python eval/eval_calibration.py
  .venv/bin/python eval/eval_calibration.py --llm-endpoint http://127.0.0.1:8080/v1 --n 400
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "faffabout.duckdb"
BASELINE = ROOT / "baseline"
OUT = ROOT / "eval"

LADDER = ["selected", "cloned", "expressed", "soluble", "purified",
          "crystallised", "diffracting", "structure", "deposited"]
RE_P = re.compile(r"([01]?\.\d+|[01])\s*$|:\s*([01]?\.\d+)")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).sum()
                     for b in range(bins) if (idx == b).sum()) / len(y))


def metrics(y, p) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {"n": int(len(y)), "base_rate": float(y.mean()),
           "brier": float(brier_score_loss(y, p)), "ece": ece(y, p)}
    out["auroc"] = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else None
    return out


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                         "mean_predicted": round(float(p[m].mean()), 3),
                         "observed": round(float(y[m].mean()), 3),
                         "gap": round(float(y[m].mean() - p[m].mean()), 3)})
    return pd.DataFrame(rows)


def gbm_predictions(con, split_col: str, test_label: str) -> pd.DataFrame:
    """Score the held-out rows with the saved boosters."""
    import lightgbm as lgb
    import sys
    sys.path.insert(0, str(BASELINE))
    import gbm_baseline as gbm

    df = gbm.load(con, split_col)
    X = gbm.prep(df, gbm.PREDICTION_TIME)
    out = []
    for g in range(8):
        mp = BASELINE / "models" / f"gate_{g}.txt"
        if not mp.exists():
            continue
        m = (df.gate == g) & (df.split == test_label)
        if not m.sum():
            continue
        b = lgb.Booster(model_file=str(mp))
        sub = df[m].copy()
        sub["p_gbm"] = b.predict(X[m][b.feature_name()])
        sub["y"] = (sub.label == "cleared").astype(int)
        out.append(sub[["target_id", "gate", "y", "p_gbm", "max_stage"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def bottleneck_top1(df: pd.DataFrame, pcol: str) -> float:
    """Does the predicted weakest gate match where the target actually stopped?"""
    hit = tot = 0
    for tid, g in df.groupby("target_id"):
        if len(g) < 2 or g.max_stage.iloc[0] >= 8:
            continue
        predicted = int(g.loc[g[pcol].idxmin(), "gate"])
        actual = int(g.max_stage.iloc[0])
        hit += predicted == actual
        tot += 1
    return hit / tot if tot else float("nan")


def llm_predictions(rows: list[dict], endpoint: str, model: str, n: int) -> list[float]:
    """Ask a served adapter for its probability on each held-out prompt.

    The model field must exactly match the server's resolved path (check /v1/models):
    a mismatch 404s as an opaque hub-lookup error rather than a clear one.
    """
    import requests
    out = []
    for i, r in enumerate(rows[:n]):
        body = {"model": model, "messages": r["messages"][:2], "max_tokens": 120, "temperature": 0.0}
        try:
            resp = requests.post(f"{endpoint}/chat/completions", json=body, timeout=120)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            m = re.search(r"([01]?\.\d+)", text)
            out.append(float(m.group(1)) if m else float("nan"))
        except Exception as e:  # noqa: BLE001
            print(f"  request {i} failed: {type(e).__name__}")
            out.append(float("nan"))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{min(n, len(rows))}", end="\r", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-endpoint", default=None, help="e.g. http://127.0.0.1:8080/v1")
    ap.add_argument("--llm-model", default=None, help="must match the server's resolved path exactly")
    ap.add_argument("--n", type=int, default=400, help="held-out prompts to send to the LLM")
    args = ap.parse_args()

    con = duckdb.connect(str(DB), read_only=True)
    report: dict = {}

    for split_col, test_label, tag in (("split_cluster", "test", "cluster"),
                                       ("split_temporal", "test", "temporal")):
        pred = gbm_predictions(con, split_col, test_label)
        if pred.empty:
            continue
        print(f"\n=== {tag} split, GBM ===")
        per_gate = {}
        for g, sub in pred.groupby("gate"):
            per_gate[int(g)] = {"gate_name": f"{LADDER[g]} -> {LADDER[g+1]}",
                                **metrics(sub.y, sub.p_gbm)}
        rows = [{"gate": g, "transition": v["gate_name"], "n": v["n"],
                 "AUROC": round(v["auroc"], 4) if v["auroc"] else None,
                 "Brier": round(v["brier"], 4), "ECE": round(v["ece"], 4)}
                for g, v in sorted(per_gate.items())]
        print(pd.DataFrame(rows).to_string(index=False))
        overall = metrics(pred.y, pred.p_gbm)
        b1 = bottleneck_top1(pred, "p_gbm")
        print(f"pooled: Brier {overall['brier']:.4f}  ECE {overall['ece']:.4f}  "
              f"AUROC {overall['auroc']:.4f}   bottleneck top-1 {b1:.3f}")
        print("\nreliability (GBM, pooled):")
        print(reliability(pred.y.values, pred.p_gbm.values).to_string(index=False))
        report[tag] = {"gbm": {"per_gate": per_gate, "pooled": overall, "bottleneck_top1": b1}}

    if args.llm_endpoint:
        test_path = ROOT / "data" / "sft" / "test.jsonl"
        meta_path = ROOT / "data" / "sft" / "test_tasks.jsonl"
        recs = [json.loads(l) for l in test_path.open()]
        meta = [json.loads(l) for l in meta_path.open() if l.strip()]
        pairs = [(r, m) for r, m in zip(recs, meta)
                 if m["task"] == "gate_judgement" and m.get("label") in ("cleared", "failed")]
        print(f"\n=== LLM on {min(args.n, len(pairs))} held-out gate judgements ===")
        p_llm = llm_predictions([r for r, _ in pairs], args.llm_endpoint,
                                args.llm_model or "faffabout", args.n)
        y = np.array([1 if m["label"] == "cleared" else 0 for _, m in pairs[:len(p_llm)]])
        p = np.array(p_llm)
        ok = ~np.isnan(p)
        print(f"parsed a probability from {ok.sum()}/{len(p)} responses")
        if ok.sum() > 20:
            llm = metrics(y[ok], p[ok])
            print(f"LLM: Brier {llm['brier']:.4f}  ECE {llm['ece']:.4f}  AUROC {llm['auroc']:.4f}")
            print("\nreliability (LLM):")
            print(reliability(y[ok], p[ok]).to_string(index=False))
            # the number that justifies the fine-tune
            gbm_pooled = report.get("cluster", {}).get("gbm", {}).get("pooled")
            if gbm_pooled:
                print(f"\nGBM delta (LLM minus GBM, lower Brier is better): "
                      f"Brier {llm['brier'] - gbm_pooled['brier']:+.4f}  "
                      f"ECE {llm['ece'] - gbm_pooled['ece']:+.4f}  "
                      f"AUROC {llm['auroc'] - gbm_pooled['auroc']:+.4f}")
                print("If the LLM does not win on any of these, its value is the narrative,\n"
                      "and eval_generative.py is where that has to be demonstrated.")
            report["llm"] = llm
        else:
            print("too few parsable responses to score")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {OUT / 'calibration.json'}")


if __name__ == "__main__":
    main()
