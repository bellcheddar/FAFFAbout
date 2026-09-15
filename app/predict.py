"""predict.py: the forecast, assembled from the gradient-booster and the archive.

The division of labour the specification insists on (section 7.4), enforced here by
construction rather than by convention:

  every NUMBER            comes from the GBM boosters or from DuckDB
  the `narrative` field   is the only thing a language model writes

So this module never calls the model, and `llm.py` never computes a probability. If a
narrative is unavailable the forecast is still complete; it simply has no prose.

Two booster sets are loaded. The headline set (37 prediction-time features) excludes host,
tag and protease, because in the archive their mere presence tracks progression: at the
first gate, host is missing for 96.2% of failures and 27.3% of successes. The
`declared_` set adds them and was trained only on rows that already declare a protocol,
where presence is constant and only the choice varies. So:

  the user declares no protocol  ->  headline boosters, no counterfactuals offered
  the user declares a host/tag   ->  declared_ boosters, counterfactuals available

That is the only honest way to answer "what if I switch host?" from this archive.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import features_seq as fs          # noqa: E402

# Serving uses the DuckDB-backed lookup, not the in-memory dumps: see app/taxo.py for the
# measurement that motivated it. The batch class is the fallback for a checkout that has
# not built the table yet.
try:
    import taxo as _taxo
    _USE_TAXO_TABLE = _taxo.available()
except Exception:  # noqa: BLE001
    _USE_TAXO_TABLE = False
if not _USE_TAXO_TABLE:
    from taxonomy import Taxonomy  # noqa: E402

MODELS = ROOT / "baseline" / "models"
N_GATES = 8

LADDERS = {
    "xray": ["selected", "cloned", "expressed", "soluble", "purified",
             "crystallised", "diffracting", "structure", "deposited"],
    "nmr": ["selected", "cloned", "expressed", "soluble", "purified",
            "labelled", "assigned", "structure", "deposited"],
    "em": ["selected", "cloned", "expressed", "soluble", "purified",
           "vitrified", "classified", "structure", "deposited"],
}
GATE_ACTION = [
    "cloning the gene into an expression vector",
    "getting detectable expression",
    "getting soluble protein",
    "purifying it to homogeneity",
    "obtaining crystals",
    "obtaining useful diffraction",
    "solving the structure",
    "depositing the structure",
]

# The levers the interface offers, and the archive value each maps to.
HOSTS = {"bl21": "ecoli", "rosetta": "ecoli", "arctic": "ecoli",
         "cellfree": "cell_free", "sf9": "insect", "hek": "mammalian", "pichia": "yeast"}
# `host` collapses to five archive values, so BL21 at 37 C and Arctic Express at 12 C are
# the SAME value and cannot differ on that feature alone. cold_shock, mined from the same
# protocol text, is what separates them; without this mapping the two choices returned
# byte-identical forecasts and the interface's headline counterfactual was a no-op.
COLD_SHOCK_HOSTS = {"arctic"}
TAGS = {"his": "his", "mbp": "mbp", "sumo": "sumo", "gst": "gst", "strep": "strep", "none": ""}
PROTEASES = {"tev": "tev", "3c": "prescission", "thr": "thrombin", "none": ""}

_TAX = None
_BOOSTERS: dict[str, dict] = {}
_SCHEMAS: dict[str, dict] = {}


class _BatchTax:
    """Adapter so the rest of this module does not care which backend answered."""

    def __init__(self):
        self._t = Taxonomy()

    def classify(self, taxid=None, organism: str = "") -> dict:
        return self._t.classify(taxid, organism)


def taxonomy():
    global _TAX
    if _USE_TAXO_TABLE:
        return _taxo
    if _TAX is None:
        _TAX = _BatchTax()
    return _TAX


def boosters(prefix: str = "") -> dict:
    """Load and cache one booster per gate, plus the terminal model."""
    import lightgbm as lgb
    if prefix in _BOOSTERS:
        return _BOOSTERS[prefix]
    out = {}
    for g in range(N_GATES):
        p = MODELS / f"{prefix}gate_{g}.txt"
        if p.exists():
            out[g] = lgb.Booster(model_file=str(p))
    tp = MODELS / f"{prefix}terminal.txt"
    if tp.exists():
        out["terminal"] = lgb.Booster(model_file=str(tp))
    if not out:
        raise FileNotFoundError(f"no boosters with prefix {prefix!r} in {MODELS}; "
                                "run baseline/gbm_baseline.py")
    _BOOSTERS[prefix] = out
    return out


@dataclass
class Choices:
    method: str = "xray"
    centre: str = "JCSG"
    host: str = ""            # "" means undeclared
    tag: str = ""
    protease: str = ""
    codon_optimised: bool = False
    autoinduction: bool = False
    iptg: bool = False

    @property
    def declares_protocol(self) -> bool:
        return bool(self.host or self.tag or self.protease)

    @property
    def cold_shock(self) -> bool:
        """Low-temperature induction, which in this archive is the Arctic Express route."""
        return self.host in COLD_SHOCK_HOSTS


def sequence_features(sequence: str) -> dict:
    counts = fs.counts_matrix([sequence])
    dis, dn, dc, di = fs.disorder_fractions(sequence)
    return {
        "seq_len": len(sequence),
        "pi": float(fs.isoelectric_point(counts)[0]),
        "gravy": float(fs.gravy(counts)[0]),
        "net_charge_ph7": float(fs.net_charge(counts, 7.0)[0]),
        "aromatic_frac": float(fs.aromatic_fraction(counts)[0]),
        "cys_frac": float(fs.residue_fraction(counts, "C")[0]),
        "met_frac": float(fs.residue_fraction(counts, "M")[0]),
        "gly_frac": float(fs.residue_fraction(counts, "G")[0]),
        "pro_frac": float(fs.residue_fraction(counts, "P")[0]),
        "cys_count": int(counts[0, fs.AA_INDEX["C"]]),
        "met_count": int(counts[0, fs.AA_INDEX["M"]]),
        "tm_helices": fs.tm_helices(sequence),
        "signal_peptide": fs.has_signal_peptide(sequence),
        "disorder_frac": dis, "disorder_nterm": dn,
        "disorder_cterm": dc, "disorder_internal": di,
        "low_complexity_frac": fs.low_complexity_fraction(sequence),
    }


def feature_row(sequence: str, organism: str, taxon_id: str, ctx: dict,
                choices: Choices, gate: int) -> dict:
    tax = taxonomy().classify(taxon_id, organism)
    per_gate = ctx.get("per_gate", {}).get(gate, {})
    row = sequence_features(sequence)
    row |= {
        "superkingdom": tax["superkingdom"] or None,
        "kingdom": tax["kingdom"] or None,
        "phylum": tax["phylum"] or None,
        "centre": choices.centre or None,
        # target-definition features a novel query cannot claim
        "n_sequences": 1, "n_references": 0,
        "has_pfam": False, "has_uniprot": False,
        "archive_says_membrane": row["tm_helices"] >= 3,
        "archive_says_eukaryotic": tax["superkingdom"] == "Eukaryota",
        "archive_says_multidomain": False,
        # archive context, computed from the search rather than from cluster membership
        "n_cluster_precedents": ctx.get("n_cluster_precedents", 0),
        "n_close_precedents": ctx.get("n_close_precedents", 0),
        "closest_identity_band": ctx.get("closest_identity_band", 0.0),
        "cluster_censored_frac": ctx.get("cluster_censored_frac"),
        "cluster_deposit_rate": ctx.get("cluster_deposit_rate"),
        "n_precedents": per_gate.get("n_precedents", 0),
        "n_precedents_cleared": per_gate.get("n_precedents_cleared", 0),
        "cluster_base_rate": per_gate.get("cluster_base_rate"),
        # declared protocol
        "host": HOSTS.get(choices.host, choices.host) or None,
        "tag": TAGS.get(choices.tag, choices.tag) or None,
        "protease": PROTEASES.get(choices.protease, choices.protease) or None,
        "codon_optimised": choices.codon_optimised,
        "autoinduction": choices.autoinduction,
        "cold_shock": choices.cold_shock,
        "iptg": choices.iptg,
    }
    return row


def schema(prefix: str = "") -> dict:
    """Columns and category levels the boosters were trained on."""
    import json
    if prefix in _SCHEMAS:
        return _SCHEMAS[prefix]
    p = MODELS / f"{prefix}schema.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no {p.name} beside the boosters. Re-run baseline/gbm_baseline.py, which "
            "writes the category levels the model needs in order to score a new row.")
    _SCHEMAS[prefix] = json.loads(p.read_text())
    return _SCHEMAS[prefix]


def _frame(row: dict, booster, prefix: str) -> pd.DataFrame:
    """One row in exactly the columns and dtypes this booster was trained on.

    The category LEVELS come from the saved schema, not from the values in this row:
    LightGBM encodes a categorical by its pandas category code, so a fresh one-row frame
    would assign code 0 to whatever value it happens to hold and every prediction would
    silently be about the wrong centre or the wrong host.
    """
    sch = schema(prefix)
    names = booster.feature_name()
    cats = sch["categorical"]
    data = {}
    for n in names:
        v = row.get(n)
        if isinstance(v, bool):
            v = float(v)
        data[n] = [v]
    df = pd.DataFrame(data)
    for n in names:
        if n in cats:
            dtype = pd.CategoricalDtype(categories=cats[n])
            val = df[n].iloc[0]
            df[n] = pd.Series([val if (val in cats[n]) else None], dtype=dtype)
        else:
            df[n] = pd.to_numeric(df[n], errors="coerce").astype("float64")
    return df[names]


def conditional(sequence: str, organism: str, taxon_id: str, ctx: dict,
                choices: Choices) -> tuple[list[float], str]:
    """Probability of clearing each gate, given the one before it cleared."""
    prefix = "declared_" if choices.declares_protocol else ""
    try:
        bst = boosters(prefix)
    except FileNotFoundError:
        prefix = ""
        bst = boosters("")
    out = []
    for g in range(N_GATES):
        b = bst.get(g)
        if b is None:
            out.append(float("nan"))
            continue
        row = feature_row(sequence, organism, taxon_id, ctx, choices, g)
        p = float(b.predict(_frame(row, b, prefix))[0])
        out.append(min(0.99, max(0.01, p)))
    return out, prefix


def survival(cond: list[float]) -> list[float]:
    """Cumulative probability of still being alive at each rung, starting at 1.0."""
    out, run = [1.0], 1.0
    for c in cond:
        run *= c
        out.append(run)
    return out


def bottleneck(surv: list[float]) -> dict:
    """The gate that loses the most probability mass in absolute terms."""
    drops = [surv[i] - surv[i + 1] for i in range(len(surv) - 1)]
    i = int(np.argmax(drops))
    return {"gate": i, "drop": drops[i]}


def jeffreys_interval(cleared: int, entered: int, z: float = 1.96) -> tuple[float, float]:
    """A Jeffreys (Beta(1/2, 1/2)) interval on the precedent counts.

    The booster gives a point estimate with no notion of how much evidence sits behind it.
    This is the honest width: three precedents produce a band you cannot plan against, and
    the rig's placeholder of a fixed +/- 0.045 per censored record was invented.
    """
    from scipy.stats import beta
    if entered <= 0:
        return (0.0, 1.0)
    a, b = cleared + 0.5, entered - cleared + 0.5
    lo = float(beta.ppf(0.025, a, b)) if cleared > 0 else 0.0
    hi = float(beta.ppf(0.975, a, b)) if cleared < entered else 1.0
    return (lo, hi)


def forecast(resolved, ctx: dict, precedents: list, choices: Choices) -> dict:
    """The full POST /predict payload, minus the narrative."""
    seq = resolved.sequence
    ladder = LADDERS.get(choices.method, LADDERS["xray"])
    cond, prefix = conditional(seq, resolved.organism, resolved.taxon_id, ctx, choices)
    surv = survival(cond)
    bn = bottleneck(surv)
    g = bn["gate"]
    pg = ctx.get("per_gate", {}).get(g, {})
    lo, hi = jeffreys_interval(pg.get("n_precedents_cleared", 0) or 0,
                              pg.get("n_precedents", 0) or 0)

    feats = sequence_features(seq)
    tax = taxonomy().classify(resolved.taxon_id, resolved.organism)
    caveats = build_caveats(ctx, feats, tax, choices, resolved)

    return {
        "target": resolved.as_dict(),
        "features": {
            "length": feats["seq_len"], "pI": round(feats["pi"], 2),
            "gravy": round(feats["gravy"], 3),
            "net_charge_ph7": round(feats["net_charge_ph7"], 1),
            "cys_count": feats["cys_count"], "met_count": feats["met_count"],
            "tm_helices": feats["tm_helices"], "signal_peptide": feats["signal_peptide"],
            "disorder_frac": round(feats["disorder_frac"], 3),
            "disorder_nterm": round(feats["disorder_nterm"], 3),
            "disorder_cterm": round(feats["disorder_cterm"], 3),
            "low_complexity_frac": round(feats["low_complexity_frac"], 3),
            "superkingdom": tax["superkingdom"], "genus": tax["genus"],
        },
        "ladder": ladder,
        "method": choices.method,
        "conditional": [round(c, 4) for c in cond],
        "survival": [round(s, 4) for s in surv],
        "bottleneck": {"gate": g, "name": ladder[g], "next": ladder[g + 1],
                       "action": GATE_ACTION[g], "drop": round(bn["drop"], 4),
                       "conditional": round(cond[g], 4),
                       "interval": [round(lo, 3), round(hi, 3)]},
        "evidence": {
            "n_precedents": ctx.get("n_precedents", 0),
            "n_close": ctx.get("n_close_precedents", 0),
            "n_censored": ctx.get("n_censored", 0),
            "closest_identity": round(ctx.get("closest_identity", 0.0), 3),
            "strength": ctx.get("strength", "NONE"),
            "per_gate": {str(k): v for k, v in ctx.get("per_gate", {}).items()},
        },
        "precedents": [p.as_dict() for p in precedents[:40]],
        "counterfactuals": counterfactuals(resolved, ctx, choices, surv, g),
        "model": {"probabilities_from": f"lightgbm:{prefix or 'headline'}",
                  "narrative_from": None, "narrative": None},
        "caveats": caveats,
    }


COUNTERFACTUALS = [
    ("host", "arctic", "Host to E. coli Arctic Express, 12 C (cold induction)"),
    ("host", "bl21", "Host to E. coli BL21(DE3), 37 C"),
    ("host", "cellfree", "Host to cell-free"),
    ("host", "sf9", "Host to Sf9 insect cells"),
    ("tag", "mbp", "Tag to MBP fusion"),
    ("tag", "sumo", "Tag to SUMO"),
    ("protease", "tev", "Add a TEV cleavage site"),
    ("codon_optimised", True, "Codon optimise the gene"),
    ("autoinduction", True, "Use auto-induction medium"),
]


def counterfactuals(resolved, ctx: dict, choices: Choices,
                    base_surv: list[float], base_gate: int) -> list[dict]:
    """Re-run the model with one lever changed. Only meaningful once a protocol is declared.

    Requires the declared_ boosters, which are the only ones that take host, tag and
    protease as inputs. Without a declared protocol the honest answer is that this archive
    cannot say, so the list comes back empty with a reason attached.
    """
    if not choices.declares_protocol:
        return []
    try:
        boosters("declared_")
    except FileNotFoundError:
        return []
    out = []
    from dataclasses import replace
    for field_name, value, label in COUNTERFACTUALS:
        if getattr(choices, field_name) == value:
            continue
        alt = replace(choices, **{field_name: value})
        try:
            cond, _ = conditional(resolved.sequence, resolved.organism,
                                  resolved.taxon_id, ctx, alt)
        except Exception:  # noqa: BLE001
            continue
        s = survival(cond)
        bn = bottleneck(s)
        out.append({
            "change": f"{field_name}:{value}",
            "label": label,
            "delta_pts": round((s[base_gate + 1] - base_surv[base_gate + 1]) * 100, 1),
            "delta_deposition_pts": round((s[-1] - base_surv[-1]) * 100, 1),
            "moves_bottleneck_to": (LADDERS[choices.method][bn["gate"]]
                                    if bn["gate"] != base_gate else None),
        })
    out.sort(key=lambda d: -d["delta_pts"])
    return out[:6]


def build_caveats(ctx: dict, feats: dict, tax: dict, choices: Choices, resolved) -> list[str]:
    """Everything a reader needs in order to discount this forecast correctly."""
    c: list[str] = list(resolved.warnings)
    n, n_close = ctx.get("n_precedents", 0), ctx.get("n_close_precedents", 0)
    n_cens = ctx.get("n_censored", 0)

    if n == 0:
        c.append("No archive precedent above 30% identity. This forecast is the model's "
                 "prior for a protein of these properties, not evidence about this family.")
    elif n < 5:
        c.append(f"Only {n} usable precedent{'s' if n != 1 else ''} above 30% identity, so "
                 "the interval is wide and the base rates are unstable.")
    if n and not n_close:
        c.append("No precedent above 70% identity; the closest relatives are distant.")
    if n_cens:
        c.append(f"{n_cens} of {n + n_cens} retrieved precedents are censored, meaning their "
                 "files closed when a centre stopped reporting rather than on any result. "
                 "They are excluded from the rates and shown grey and dashed.")
    if feats["tm_helices"] >= 3:
        c.append(f"{feats['tm_helices']} predicted transmembrane helices. PSI centres "
                 "filtered for soluble prokaryotic targets, so a membrane protein is "
                 "extrapolation beyond what this archive can support.")
    if tax["superkingdom"] == "Eukaryota":
        c.append("Eukaryotic source. The archive is dominated by prokaryotic targets, so "
                 "eukaryotic predictions are weaker than the numbers imply.")
    if feats["disorder_frac"] > 0.4:
        c.append(f"{feats['disorder_frac']:.0%} of the sequence is predicted disordered, "
                 "which the archive's own records rarely annotate explicitly.")
    if not choices.declares_protocol:
        c.append("No expression host or tag declared, so no counterfactuals are offered: "
                 "in the archive the presence of a protocol is itself correlated with "
                 "progression, and the headline model deliberately excludes it.")
    c.append("PSI target selection was not random. Every number here is conditioned on a "
             "target having been chosen by a structural genomics centre between 2000 and 2017.")
    return c
