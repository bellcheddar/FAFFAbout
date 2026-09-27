#!/usr/bin/env python
"""06b_gbm_forecasts.py: an honest GBM forecast for every target, for the round 5 corpus.

From round 5 the model NARRATES the GBM's forecast instead of inventing one (CLAUDE.md
rule 5: every number comes from the GBM). So each training prompt carries the GBM's
per-gate conditionals, and those have to be the numbers the GBM would really give that
target at serving time. A booster scoring a target it was trained on gives optimistic,
overconfident numbers the model would never meet in use, so:

  p_cluster   split_cluster = 'train' targets: 5-fold OUT-OF-FOLD, grouped by cluster_id,
              so no booster scores a target whose cluster it trained on.
              valid / test / none: boosters fit on the whole training split.
  p_temporal  split_temporal = 'test' targets only: the temporal boosters
              (baseline/models/temporal_*), trained before 2014. This is what the
              test_temporal held-out set and eval_bottleneck.py use.

Early stopping inside each fit uses a cluster-grouped 10% of that fit's own training rows,
never the valid split, so valid-split forecasts are not tuned on valid targets.

Output: data/parquet/gbm_forecast.parquet, one row per (target_id, gate).

Usage
  .venv/bin/python scripts/06b_gbm_forecasts.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))
import gbm_baseline as gb  # noqa: E402

OUT = ROOT / "data" / "parquet" / "gbm_forecast.parquet"
FOLDS = 5
SEED = 42


def fit(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> lgb.Booster:
    """gb.fit_one, with early stopping on a cluster-grouped 10% of the SAME rows."""
    uniq = np.unique(groups)
    stop = set(rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False))
    va = np.isin(groups, list(stop))
    return gb.fit_one(X[~va], X[va], y[~va], y[va])


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    con = duckdb.connect(str(gb.DB), read_only=True)

    rows = gb.load(con, "split_cluster")
    cid = dict(con.execute("SELECT target_id, cluster_id FROM splits").fetchall())
    rows["cluster_id"] = rows.target_id.map(cid).fillna(-1).astype(int)
    train = rows[rows.split == "train"].reset_index(drop=True)
    X_train = gb.prep(train, gb.PREDICTION_TIME)
    # Freeze category levels from the training rows, so every prediction frame encodes a
    # categorical with the same codes the boosters learned (see gb.dump_schema).
    levels = {c: list(X_train[c].cat.categories) for c in X_train if str(X_train[c].dtype) == "category"}
    print(f"{len(train):,} training L1 rows, {train.target_id.nunique():,} targets")

    # every target at every gate, prediction-time features only
    grid = con.execute("""
        SELECT c.target_id, c.gate, s.split_cluster, s.split_temporal, s.cluster_id,
               f.* EXCLUDE (target_id, organism, seq_md5),
               c.n_precedents, c.n_precedents_cleared, c.cluster_base_rate
        FROM features_context c JOIN splits s USING (target_id)
        LEFT JOIN features f USING (target_id)
    """).df()
    Xg = gb.prep(grid, gb.PREDICTION_TIME)
    for c, lv in levels.items():
        Xg[c] = pd.Categorical(grid[c].astype("string"), categories=lv)
    Xg = Xg[X_train.columns]
    print(f"{len(grid):,} (target, gate) cells to forecast")

    # cluster folds over TRAINING targets
    train_clusters = np.unique(train.cluster_id)
    fold_of = dict(zip(rng.permutation(train_clusters), np.arange(len(train_clusters)) % FOLDS))
    grid_fold = grid.cluster_id.fillna(-1).astype(int).map(fold_of)
    is_train_target = (grid.split_cluster == "train").values
    train_fold = train.cluster_id.map(fold_of).values

    p_cluster = np.full(len(grid), np.nan)
    for g in range(8):
        tg = (train.gate == g).values
        y = (train.label[tg] == "cleared").astype(int).values
        Xt, grp, fold = X_train[tg], train.cluster_id[tg].values, train_fold[tg]
        cells = (grid.gate == g).values
        for k in range(FOLDS):
            b = fit(Xt[fold != k], y[fold != k], grp[fold != k], rng)
            m = cells & is_train_target & (grid_fold == k).values
            p_cluster[m] = b.predict(Xg[m], num_iteration=b.best_iteration)
        b = fit(Xt, y, grp, rng)
        # Held-out targets, plus training targets whose cluster has no in-loss rows at all
        # (2,596 on 2026-09-27, e.g. clusters that are wholly censored): no booster trained on
        # their cluster, so the full-training model is out-of-fold for them too.
        m = cells & (~is_train_target | grid_fold.isna().values)
        p_cluster[m] = b.predict(Xg[m], num_iteration=b.best_iteration)
        print(f"  gate {g}: {tg.sum():,} rows, {FOLDS} folds + full  ({time.time()-t0:.0f}s)")

    # temporal boosters, trained by eval_bottleneck.py under the temporal_ prefix
    p_temporal = np.full(len(grid), np.nan)
    tschema = gb.MODELS / "temporal_schema.json"
    if tschema.exists():
        import json
        sch = json.loads(tschema.read_text())
        Xt = gb.prep(grid, sch["features"])
        for c, lv in sch["categorical"].items():
            Xt[c] = pd.Categorical(grid[c].astype("string"), categories=lv)
        Xt = Xt[sch["features"]]
        tmask = (grid.split_temporal == "test").values
        for g in range(8):
            f = gb.MODELS / f"temporal_gate_{g}.txt"
            if f.exists():
                b = lgb.Booster(model_file=str(f))
                m = tmask & (grid.gate == g).values
                p_temporal[m] = b.predict(Xt[m], num_iteration=b.best_iteration or None)
    else:
        print("  no temporal boosters: run eval/eval_bottleneck.py once to train them")

    out = pd.DataFrame({"target_id": grid.target_id, "gate": grid.gate.astype(int),
                        "p_cluster": np.clip(p_cluster, 0.01, 0.99),
                        "p_temporal": np.clip(p_temporal, 0.01, 0.99)})
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(out):,} rows, p_cluster missing {np.isnan(p_cluster).sum():,}, "
          f"p_temporal present {(~np.isnan(p_temporal)).sum():,}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
