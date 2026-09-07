"""Feature-provenance tests: the guard against predicting the past from itself.

Trained on the post-hoc block as well, LightGBM scored a mean AUROC of 0.985 across the
gates and exactly 1.0000 on the terminal outcome, on a clean cluster-held-out split. The
split was never the problem; the columns were. These tests keep them apart.

Run:  .venv/bin/python -m pytest -q tests/test_baseline_features.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))

import gbm_baseline as gbm  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "features.yaml").read_text())
METRICS = ROOT / "baseline" / "metrics.json"
MODELS = ROOT / "baseline" / "models"


def test_the_three_blocks_are_disjoint():
    pt = set(gbm.PREDICTION_TIME)
    dp = set(gbm.DECLARED_PROTOCOL)
    ph = set(gbm.POST_HOC)
    assert pt & ph == set(), pt & ph
    assert pt & dp == set(), pt & dp
    assert dp & ph == set(), dp & ph


def test_no_feature_is_listed_twice_within_a_block():
    for block in ("prediction_time", "declared_protocol", "post_hoc"):
        names = gbm.feature_names(block)
        assert len(names) == len(set(names)), block


@pytest.mark.parametrize("banned", [
    "has_pdb_ref",       # a deposition reference is the answer itself
    "method",            # derived from the status history: encodes the stage reached
    "n_trials",          # 16.4 for cleared against 4.1 for failed
    "any_step_failed",   # literally whether a step failed
    "expression_pct",    # the measured expression level
    "solubility_pct",
    "final_conc_mg_ml",  # a yield only exists if it was purified
    "selenomethionine",  # SeMet labelling implies the target crystallised
    "construct_start",   # the boundaries actually used, not the ones you would choose
])
def test_known_outcome_markers_are_never_prediction_time(banned):
    assert banned in gbm.POST_HOC
    assert banned not in gbm.PREDICTION_TIME
    assert banned not in gbm.DECLARED_PROTOCOL


def test_protocol_levers_are_not_in_the_headline_feature_set():
    """Their PRESENCE tracks progression: at gate 0, host is null for 96.2% of failures
    and 27.3% of successes. They are evaluated separately on rows that all have one."""
    for lever in ("host", "tag", "protease"):
        assert lever not in gbm.PREDICTION_TIME
        assert lever in gbm.DECLARED_PROTOCOL


def test_sequence_and_context_features_are_prediction_time():
    for f in ("seq_len", "pi", "gravy", "tm_helices", "disorder_frac", "superkingdom",
              "centre", "n_cluster_precedents", "cluster_base_rate"):
        assert f in gbm.PREDICTION_TIME, f


@pytest.mark.skipif(not MODELS.exists() or not any(MODELS.glob("*.txt")),
                    reason="needs the trained baseline")
def test_no_saved_model_was_trained_on_a_post_hoc_feature():
    """The strongest form of the check: read it off the boosters that exist on disk."""
    import lightgbm as lgb
    banned = set(gbm.POST_HOC)
    for path in sorted(MODELS.glob("*.txt")):
        used = set(lgb.Booster(model_file=str(path)).feature_name())
        assert not (used & banned), f"{path.name} was trained on {used & banned}"


@pytest.mark.skipif(not METRICS.exists(), reason="needs the trained baseline")
def test_headline_metrics_are_not_in_the_leaked_range():
    m = json.loads(METRICS.read_text())
    aucs = [v["auroc"] for v in m["cluster"]["gates"].values() if v["auroc"]]
    mean_auc = sum(aucs) / len(aucs)
    assert mean_auc < 0.95, (
        f"mean AUROC {mean_auc:.4f} is in the range that only leakage produces; "
        "check config/features.yaml")
    assert m["cluster"]["terminal"]["auroc"] < 0.99


@pytest.mark.skipif(not METRICS.exists(), reason="needs the trained baseline")
def test_the_leak_demonstration_still_demonstrates_the_leak():
    """A negative control: a check that has only ever passed is not a check."""
    m = json.loads(METRICS.read_text())
    if "cluster_with_post_hoc" not in m:
        pytest.skip("run with --leak-check")
    leaked = m["cluster_with_post_hoc"]["terminal"]["auroc"]
    honest = m["cluster"]["terminal"]["auroc"]
    assert leaked > honest + 0.05, (
        "adding the post-hoc block back should visibly inflate the score; if it no longer "
        "does, the block is no longer capturing what it was drawn to capture")
