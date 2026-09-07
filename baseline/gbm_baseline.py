#!/usr/bin/env python
"""gbm_baseline.py: the gate the fine-tuned model has to clear.

LightGBM on the same tabular features, predicting the L1 per-gate transition and the L2
terminal outcome, on all three splits.

Expect this to beat an 8B model on AUC. That is the normal result for tabular prediction
and it does not kill the project: it defines what the language model is for. The narrative,
the attribution, the construct recommendation and the orthologue ranking are what an 8B
model adds, and if no number can be stated for what it adds over this baseline then the
language model is decoration.

This model also ships INSIDE the app as the source of every probability, with the
fine-tuned model reasoning over its output. Asking an 8B model to emit calibrated floats
is both less accurate and less honest than asking a gradient-booster.

Outputs
  baseline/models/gate_{g}.txt       one booster per gate
  baseline/models/terminal.txt       the L2 deposited/stalled model
  baseline/metrics.json              every number, for the README and the eval comparison
  baseline/feature_importance.csv

Usage
  .venv/bin/python baseline/gbm_baseline.py
  .venv/bin/python baseline/gbm_baseline.py --split temporal
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "faffabout.duckdb"
OUT = ROOT / "baseline"
MODELS = OUT / "models"

LADDER = ["selected", "cloned", "expressed", "soluble", "purified",
          "crystallised", "diffracting", "structure", "deposited"]

# Feature provenance comes from config/features.yaml, which is the file that stops this
# script predicting the past from itself. See the long comment at the top of that file:
# trained on the post-hoc block as well, this model scored a mean AUROC of 0.985 and
# exactly 1.0000 on the terminal outcome, because `has_pdb_ref` is the answer.
FEATURE_CFG = yaml.safe_load((ROOT / "config" / "features.yaml").read_text())

CATEGORICAL = {"centre", "superkingdom", "kingdom", "phylum", "host", "tag", "protease",
               "target_construct_type", "construct_type", "method"}
BOOLEAN = {"signal_peptide", "has_pfam", "has_uniprot", "has_pdb_ref", "archive_says_membrane",
           "archive_says_eukaryotic", "archive_says_multidomain", "selenomethionine",
           "autoinduction", "codon_optimised", "refolding", "detergent", "any_step_failed"}


def feature_names(which: str = "prediction_time") -> list[str]:
    """Flatten one block of config/features.yaml into a column list."""
    blocks = FEATURE_CFG[which]
    return [c for group in blocks.values() for c in group]


PREDICTION_TIME = feature_names("prediction_time")
DECLARED_PROTOCOL = feature_names("declared_protocol")
POST_HOC = feature_names("post_hoc")


PARAMS = dict(
    objective="binary", metric="binary_logloss", learning_rate=0.05, num_leaves=63,
    min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, verbose=-1, num_threads=8, seed=42,
)
N_ROUNDS = 600
EARLY_STOP = 50


def load(con: duckdb.DuckDBPyConnection, split_col: str) -> pd.DataFrame:
    """L1 rows joined to every feature, with the split label attached."""
    return con.execute(f"""
        SELECT l.target_id, l.gate, l.label, l.max_stage, l.method,
               s.{split_col} AS split,
               f.* EXCLUDE (target_id, organism, seq_md5),
               c.n_precedents, c.n_precedents_cleared, c.cluster_base_rate
        FROM labels_l1 l
        JOIN splits s USING (target_id)
        LEFT JOIN features f USING (target_id)
        LEFT JOIN features_context c ON c.target_id = l.target_id AND c.gate = l.gate
        WHERE l.in_loss
    """).df()


def prep(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for c in columns:
        if c not in df:
            continue
        if c in CATEGORICAL:
            x[c] = df[c].astype("category")
        elif c in BOOLEAN:
            x[c] = df[c].astype("float32")
        else:
            x[c] = pd.to_numeric(df[c], errors="coerce")
    return x


def fit_one(tr: pd.DataFrame, va: pd.DataFrame, y_tr, y_va) -> lgb.Booster:
    dtr = lgb.Dataset(tr, label=y_tr)
    dva = lgb.Dataset(va, label=y_va, reference=dtr)
    return lgb.train(PARAMS, dtr, num_boost_round=N_ROUNDS, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error, 10 bins. The number to quote."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            tot += abs(y[m].mean() - p[m].mean()) * m.sum()
    return float(tot / len(y))


def evaluate(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = {"n": int(len(y)), "positive_rate": float(y.mean()),
           "brier": float(brier_score_loss(y, p)), "ece": ece(y, p)}
    out["auroc"] = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else None
    return out


def run(con, split_col: str, test_label: str, tag: str, columns: list[str],
        save: bool = False, require_protocol: bool = False) -> dict:
    df = load(con, split_col)
    if require_protocol:
        # restrict to rows that already declare a protocol, so its PRESENCE is constant
        # and only the choice of host, tag and protease varies
        df = df[df.host.notna()].reset_index(drop=True)
    X = prep(df, columns)
    results = {"split": tag, "n_features": X.shape[1], "n_rows": int(len(df)), "gates": {}}
    importances: list[pd.DataFrame] = []
    MODELS.mkdir(parents=True, exist_ok=True)

    for g in range(8):
        m = df.gate == g
        sub, xs = df[m], X[m]
        y = (sub.label == "cleared").astype(int).values
        tr = (sub.split == "train").values
        va = (sub.split == "valid").values if "valid" in set(sub.split) else tr
        te = (sub.split == test_label).values
        if tr.sum() < 200 or te.sum() < 50 or len(np.unique(y[tr])) < 2:
            continue
        if va.sum() < 50:
            va = tr
        booster = fit_one(xs[tr], xs[va], y[tr], y[va])
        p = booster.predict(xs[te], num_iteration=booster.best_iteration)
        results["gates"][g] = {"gate_name": f"{LADDER[g]} -> {LADDER[g+1]}",
                               "best_iteration": booster.best_iteration, **evaluate(y[te], p)}
        if save:
            booster.save_model(str(MODELS / f"gate_{g}.txt"))
            imp = pd.DataFrame({"feature": booster.feature_name(),
                                "gain": booster.feature_importance("gain"), "gate": g})
            importances.append(imp)

    # L2 terminal outcome, one row per uncensored target
    t = df[df.gate == 0].copy()
    y2 = (t.max_stage >= 8).astype(int).values
    x2 = X[df.gate == 0]
    tr = (t.split == "train").values
    va = (t.split == "valid").values if (t.split == "valid").sum() > 50 else tr
    te = (t.split == test_label).values
    if tr.sum() > 200 and te.sum() > 50 and len(np.unique(y2[tr])) > 1:
        b2 = fit_one(x2[tr], x2[va], y2[tr], y2[va])
        p2 = b2.predict(x2[te], num_iteration=b2.best_iteration)
        results["terminal"] = {"best_iteration": b2.best_iteration, **evaluate(y2[te], p2)}
        if save:
            b2.save_model(str(MODELS / "terminal.txt"))

    if importances and save:
        (pd.concat(importances).groupby("feature").gain.sum().sort_values(ascending=False)
         .to_csv(OUT / "feature_importance.csv", header=["total_gain"]))
    return results


def show(res: dict) -> None:
    print(f"\n=== {res['split']} split ===")
    print(f"({res['n_features']} features)")
    rows = [{"gate": g, "transition": v["gate_name"], "n": v["n"],
             "base_rate": round(v["positive_rate"], 3), "AUROC": round(v["auroc"], 4) if v["auroc"] else None,
             "Brier": round(v["brier"], 4), "ECE": round(v["ece"], 4), "iters": v["best_iteration"]}
            for g, v in sorted(res["gates"].items())]
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
        aucs = [r["AUROC"] for r in rows if r["AUROC"]]
        print(f"mean AUROC across gates: {np.mean(aucs):.4f}")
    if "terminal" in res:
        t = res["terminal"]
        print(f"terminal (deposited): n={t['n']:,} base={t['positive_rate']:.4f} "
              f"AUROC={t['auroc']:.4f} Brier={t['brier']:.4f} ECE={t['ece']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["cluster", "temporal", "all"], default="all")
    ap.add_argument("--leak-check", action="store_true",
                    help="also train on the post-hoc block, to quantify what it gives away")
    args = ap.parse_args()
    con = duckdb.connect(str(DB), read_only=True)
    out = {}
    print(f"prediction-time features: {len(PREDICTION_TIME)}  "
          f"(post-hoc block held out: {len(POST_HOC)})")
    if args.split in ("cluster", "all"):
        out["cluster"] = run(con, "split_cluster", "test", "cluster", PREDICTION_TIME, save=True)
        show(out["cluster"])
    if args.split in ("temporal", "all"):
        out["temporal"] = run(con, "split_temporal", "test", "temporal", PREDICTION_TIME)
        show(out["temporal"])
    if args.split in ("cluster", "all"):
        out["declared_protocol"] = run(
            con, "split_cluster", "test",
            "cluster, rows with a declared protocol, host/tag/protease as inputs",
            PREDICTION_TIME + DECLARED_PROTOCOL, require_protocol=True)
        show(out["declared_protocol"])
    if args.leak_check:
        out["cluster_with_post_hoc"] = run(con, "split_cluster", "test",
                                           "cluster + POST-HOC (leak demonstration)",
                                           PREDICTION_TIME + DECLARED_PROTOCOL + POST_HOC)
        show(out["cluster_with_post_hoc"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'metrics.json'}")
    if (OUT / "feature_importance.csv").exists():
        imp = pd.read_csv(OUT / "feature_importance.csv").head(15)
        print("\ntop features by total gain:")
        print(imp.to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
