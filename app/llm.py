"""llm.py: the narrative, and nothing else.

This module writes prose. It never produces a number that reaches the interface: every
probability, count and interval in the payload comes from the boosters or from DuckDB
before this is called. The prompt it builds mirrors the SFT corpus (scripts/07_build_sft.py)
so the fine-tuned model sees at inference the shape it was trained on.

Two rules enforced here rather than hoped for:

  * the prompt carries the precedent identifiers, and `sanitise` strips any identifier from
    the reply that was not in the prompt. A hallucinated precedent ID is the project's
    automatic-fail condition, so the serving path refuses to pass one through.
  * any number the model writes is ignored. The interface renders the payload's numbers;
    the narrative is prose beside them, so a disagreement is visible rather than silently
    authoritative.

Served by `mlx_lm.server` on 127.0.0.1:8080. The `model` field must match the path the
server resolved, which is why it is read from /v1/models rather than assumed: a mismatch
returns an opaque hub-lookup 404 rather than a clear error.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = os.environ.get("FAFFABOUT_LLM", "http://127.0.0.1:8080/v1")
# "default_model" is the only name that attaches the adapter. mlx_lm/server.py:316 registers
# --adapter-path under that literal key, and line 389 looks the adapter up BY THE REQUESTED
# MODEL NAME, so naming the base model resolves adapter_path=None and loads a second,
# un-adapted copy. On 2026-09-16 that served plain Llama for a whole evaluation while every
# name-based check passed, because the name was genuinely correct. Verify by OUTPUT, never
# by identity: the adapted model is terse and in-register, the base model opens with
# "I can simulate a protein attrition forecast...".
MODEL_NAME = os.environ.get("FAFFABOUT_LLM_MODEL", "default_model")
TIMEOUT = float(os.environ.get("FAFFABOUT_LLM_TIMEOUT", "45"))
MAX_TOKENS = 320

SYSTEM = (
    "You are FAFFAbout, an attrition forecaster for structural biology. You judge where a "
    "protein production attempt is likely to stop, using the Protein Structure Initiative "
    "TargetTrack archive of 335,771 real attempts as precedent. You are careful about "
    "censored records, which are targets whose file closed when a centre's funding ended "
    "rather than because the science failed: they are evidence of nothing and you say so. "
    "You never invent a target identifier, a PDB code or a precedent that is not in the "
    "context you were given."
)

_MODELS_CACHE: list[str] | None = None


def available() -> bool:
    return bool(models())


def models() -> list[str]:
    global _MODELS_CACHE
    if _MODELS_CACHE is not None:
        return _MODELS_CACHE
    try:
        r = requests.get(f"{ENDPOINT}/models", timeout=3)
        r.raise_for_status()
        _MODELS_CACHE = [m["id"] for m in r.json().get("data", [])]
    except Exception:  # noqa: BLE001
        _MODELS_CACHE = []
    return _MODELS_CACHE


def resolved_model() -> str:
    """The model we intend to narrate with, never whatever happens to be listed first.

    An earlier version returned `models()[0]`, which looks reasonable and is wrong: on a
    cold server `GET /v1/models` answers 200 immediately, BEFORE the requested model has
    loaded, listing mlx-lm's built-in default (Qwen2.5-7B-Instruct-4bit). Taking the first
    entry therefore names the default, and passing that back as the `model` field makes the
    server load and serve it. On 2026-09-16 that produced 24 narratives from a model which
    had never seen this corpus, and they would have scored a flatteringly LOW template echo.

    So prefer the configured name, and fall back to a listed entry only when nothing is
    configured. `available()` no longer treats a bare model list as readiness.
    """
    if MODEL_NAME:
        return MODEL_NAME
    m = models()
    return m[0] if m else "faffabout"


def serving_intended_model() -> bool:
    """Is the server actually offering the model we were told to use?

    A 200 from /v1/models is not readiness and the list is not identity. Callers that care
    which weights answered (every evaluation, and any caller comparing rounds) should gate
    on this rather than on `available()`.
    """
    if not MODEL_NAME:
        return bool(models())
    return any(MODEL_NAME in m for m in models())


def build_prompt(payload: dict) -> str:
    """The prompt shape the model was TRAINED on, not one invented here.

    scripts/07_build_sft.py's pipeline_forecast task gives the model target features,
    archive evidence and precedents, and asks it to produce the gate-by-gate outlook and
    name the bottleneck itself. It is never shown a precomputed forecast.

    An earlier version of this function handed over a "Computed forecast:" block of GBM
    numbers and told the model they were "not yours to change" - a distribution appearing
    in no training example. The model would have met it cold at serving time. The numbers
    the model then writes are discarded by `strip_generated_numbers`, because the GBM's
    figures are rendered beside the prose and two disagreeing sets of numbers on one page
    is worse than one.
    """
    t, f, ev = payload["target"], payload["features"], payload["evidence"]
    ladder = payload["ladder"]

    bits = [f"length: {f['length']} residues"]
    if t.get("organism"):
        bits.append(f"organism: {t['organism']}")
    if f.get("superkingdom"):
        bits.append(f"kingdom: {f['superkingdom']}")
    bits += [f"pI: {f['pI']}", f"GRAVY: {f['gravy']}",
             f"net charge at pH 7: {f['net_charge_ph7']:+}",
             f"cysteines: {f['cys_count']}",
             f"predicted TM helices: {f['tm_helices']}"]
    if f.get("signal_peptide"):
        bits.append("signal peptide predicted")
    bits.append(f"disorder: {f['disorder_frac']:.0%} overall, {f['disorder_nterm']:.0%} "
                f"N-terminal, {f['disorder_cterm']:.0%} C-terminal")
    bits.append(f"low-complexity: {f['low_complexity_frac']:.0%}")

    strength = ev.get("strength", "NONE")
    evidence = (f"  {ev['n_precedents']} uncensored precedents in the 30% cluster, "
                f"{ev['n_close']} above 70% identity")
    cens = payload.get("evidence", {}).get("n_censored", 0)
    total = ev["n_precedents"] + cens
    if total:
        evidence += f"\n  {cens / total:.0%} of this cluster's records are censored"
    evidence += f"\n  evidence strength: {strength}"

    rows = ["  target              centre    organism                        reached          note"]
    for p_ in payload["precedents"][:8]:
        stage = ladder[p_["max_stage"]] if p_.get("max_stage") is not None else "unknown"
        note = "censored at centre closure" if p_.get("censored") else ""
        rows.append(f"  {p_['target_id']:<19} {p_['centre']:<9} "
                    f"{(p_.get('organism') or ''):<31} {stage:<16} {note}".rstrip())

    return ("Forecast the whole pipeline for this target.\n\n"
            "Target features:\n" + "\n".join(f"  {b}" for b in bits) + "\n\n"
            f"Archive evidence:\n{evidence}"
            + (f"\n\nPrecedents:\n" + "\n".join(rows) if len(rows) > 1 else ""))


GATE_LINE = re.compile(r"^\s*\w[\w ]*->[\w ]*:\s*[0-9.]+\s*conditional.*$", re.M)
PROB_LINE = re.compile(r"^\s*(Gate-by-gate outlook:|Probability of [^:]+:\s*[0-9.]+).*$", re.M)


def strip_generated_numbers(text: str) -> tuple[str, bool]:
    """Remove the model's own probability table, keeping its reasoning.

    Spec 7.4: every number comes from the GBM or DuckDB, and the narrative is the only
    thing the model writes. The SFT task trains it to emit a full gate vector, so it will;
    those digits must not reach a reader beside the GBM's own figures.
    """
    cleaned = PROB_LINE.sub("", GATE_LINE.sub("", text))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, cleaned != text.strip()


IDENT_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]{1,12}-[A-Za-z0-9_.]{1,20}\b")
PDB_RE = re.compile(r"\b[1-9][A-Za-z0-9]{3}\b")


def sanitise(text: str, prompt: str, payload: dict) -> tuple[str, list[str]]:
    """Remove any identifier the model was not given. Returns (text, removals).

    A hallucinated precedent identifier is the automatic-fail condition in
    eval/eval_generative.py, so the serving path will not pass one to a reader even if the
    model produces it. The removal is reported rather than hidden.
    """
    allowed = set(IDENT_RE.findall(prompt)) | {p["target_id"] for p in payload.get("precedents", [])}
    allowed_pdb = set(PDB_RE.findall(prompt))
    removed: list[str] = []
    out = text
    for ident in set(IDENT_RE.findall(text)):
        if ident not in allowed and "-" in ident and any(c.isdigit() for c in ident):
            removed.append(ident)
            out = out.replace(ident, "[identifier removed: not in the retrieved context]")
    for pdb in set(PDB_RE.findall(text)):
        if pdb not in allowed_pdb and not pdb.isdigit():
            removed.append(pdb)
            out = out.replace(pdb, "[PDB code removed: not in the retrieved context]")
    return out, removed


def narrate(payload: dict) -> str | None:
    """The narrative, or None if no model is serving."""
    if not available():
        return None
    prompt = build_prompt(payload)
    body = {"model": resolved_model(),
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS, "temperature": 0.2}
    try:
        r = requests.post(f"{ENDPOINT}/chat/completions", json=body, timeout=TIMEOUT)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:  # noqa: BLE001
        return None
    clean, stripped = strip_generated_numbers(text)
    if stripped:
        payload.setdefault("caveats", []).append(
            "The model's own probability estimates were removed from the interpretation: "
            "every figure shown is computed by the gradient-boosted model, not written by "
            "the language model.")
    clean, removed = sanitise(clean, prompt, payload)
    if removed:
        payload.setdefault("caveats", []).append(
            f"The model named {len(removed)} identifier(s) that were not in its retrieved "
            "context; they have been removed. This is a model fault worth reporting.")
    return clean
