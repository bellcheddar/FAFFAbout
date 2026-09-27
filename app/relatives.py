"""relatives.py: what close relatives in the archive actually tried, and how it went.

This replaces the fine-tuned narrative (2026-09-27). Round 5's prose passed every automated
check and was judged useful in 0 of 10 cases, because it was trained on template sentences
and so could only restate the forecast. What a scientist can act on is already in the
archive: the constructs, hosts and tags that close relatives were given, how far each
attempt got, and the reason the centre recorded when it stopped. Every line here is read
from the archive; nothing is generated.

Coverage varies by centre (MCSG records no construct types, for example), so every field
is optional and the text says only what was recorded.
"""
from __future__ import annotations

import re
from collections import Counter

LADDER = ["selected", "cloned", "expressed", "soluble", "purified",
          "crystallised", "diffracting", "structure", "deposited"]

MAX_RELATIVES = 6          # the closest relatives shown in detail
MAX_TRIALS = 3             # trials listed per relative
# The archive's protocol mining leaves lower-case codes; show what a scientist would write.
HOST_LABEL = {"ecoli": "E. coli", "insect": "insect cells", "cell_free": "cell-free",
              "mammalian": "mammalian cells", "yeast": "yeast"}
TAG_LABEL = {"his": "His", "gst": "GST", "mbp": "MBP", "sumo": "SUMO", "strep": "Strep"}
ADMIN_STOPS = {"other", "duplicate target found", "pdb duplicate found", "structure successful"}

_START = re.compile(r"Sequence start:\s*(\d+)")
_END = re.compile(r"Sequence end:\s*(\d+)")
_EXPR = re.compile(r"Expression level\s*:\s*(\d+)\s*%")
_SOL = re.compile(r"Solubility level\s*:\s*(\d+)\s*%")
_CONC = re.compile(r"Final concentration:\s*([\d.]+)\s*mg/mL")


def _first(rx: re.Pattern, text: str | None) -> str | None:
    m = rx.search(text or "")
    return m.group(1) if m else None


def _construct(ctype: str | None, notes: str | None) -> str:
    kind = (ctype or "").split("|")[-1].strip()
    s, e = _first(_START, notes), _first(_END, notes)
    span = f"residues {s}-{e}" if s and e else ""
    return ", ".join(x for x in (kind, span) if x)


def _stop(status: str | None, remark: str | None) -> str:
    s = (status or "").strip()
    r = " ".join((remark or "").split())
    if len(r) > 140:
        r = r[:137].rstrip() + "..."
    if s and r:
        return f"{s}: {r}"
    return s or r


def trials_for(con, target_ids: list[str]) -> dict[str, list[dict]]:
    """Every trial of these targets, with the furthest stage that trial itself reached."""
    if not target_ids:
        return {}
    rows = con.execute("""
        SELECT t.target_id, t.trial_id, t.construct_type, t.stop_status, t.stop_remark,
               t.free_text_notes, max(h.stage_ord) AS best
        FROM trials t
        LEFT JOIN status_history_canon h ON h.target_id = t.target_id AND h.trial_id = t.trial_id
        WHERE t.target_id IN (SELECT unnest(?))
        GROUP BY ALL
    """, [target_ids]).fetchall()
    out: dict[str, list[dict]] = {}
    for tid, trial, ctype, stop, remark, notes, best in rows:
        out.setdefault(tid, []).append({
            "trial": trial,
            "reached": LADDER[best] if best is not None else None,
            "best": best,
            "construct": _construct(ctype, notes),
            "construct_kind": (ctype or "").split("|")[-1].strip(),
            "expression_pct": _first(_EXPR, notes),
            "solubility_pct": _first(_SOL, notes),
            "final_mg_ml": _first(_CONC, notes),
            "stopped": _stop(stop, remark),
            "stop_status": (stop or "").strip().lower(),
        })
    for tid in out:
        out[tid].sort(key=lambda t: (-(t["best"] if t["best"] is not None else -1), t["trial"]))
    return out


def facts(relatives: list[dict], gate: int) -> list[str]:
    """Plain statements about the bottleneck step, each true of the relatives listed.

    Censored relatives are excluded from every count (rule 2: a file closed at centre
    shutdown is evidence of nothing) and reported separately.
    """
    live = [r for r in relatives if not r["censored"] and r["max_stage"] is not None]
    out: list[str] = []
    at, nxt = LADDER[gate], LADDER[gate + 1]

    entered = [r for r in live if r["max_stage"] >= gate]
    if entered:
        cleared, n = sum(r["max_stage"] > gate for r in entered), len(entered)
        if n == 1:
            out.append(f"The one close relative that reached {at} "
                       f"{'got through to ' + nxt if cleared else 'did not get through to ' + nxt}.")
        elif cleared == 0:
            out.append(f"None of the {n} close relatives that reached {at} got through to {nxt}.")
        else:
            out.append(f"{cleared} of the {n} close relatives that reached {at} got through to {nxt}.")
    elif live:
        furthest = max(live, key=lambda r: r["max_stage"])
        out.append(f"None of the close relatives reached {at}; the furthest, {furthest['target_id']}, "
                   f"stopped at {LADDER[furthest['max_stage']]}.")

    reasons = Counter(
        t["stop_status"] for r in live if r["max_stage"] == gate
        for t in r["trials"] if t["stop_status"] and t["stop_status"] not in ADMIN_STOPS)
    if reasons:
        (why, n), = reasons.most_common(1)
        out.append(f"Where relatives stopped at {at}, the reason most often recorded was "
                   f"\"{why}\" ({n} trial{'s' if n != 1 else ''}).")

    kinds: dict[str, int] = {}
    for r in live:
        for t in r["trials"]:
            k = t["construct_kind"]
            if k and t["best"] is not None:
                kinds[k] = max(kinds.get(k, -1), t["best"])
    full = kinds.get("full length ORF")
    trunc = max((v for k, v in kinds.items() if k != "full length ORF"), default=None)
    if full is not None and trunc is not None and full != trunc:
        better, worse = ("truncated or domain constructs", "full-length ones") if trunc > full \
            else ("full-length constructs", "truncated or domain ones")
        out.append(f"Among these relatives, {better} got further ({LADDER[max(full, trunc)]}) "
                   f"than {worse} ({LADDER[min(full, trunc)]}).")

    combos = [(r["host"], r["tag"], r["max_stage"]) for r in live if r["host"] or r["tag"]]
    if len(combos) >= 2:
        h, t, s = max(combos, key=lambda c: c[2])
        label = " with ".join(x for x in (h, t and f"{t} tag") if x)
        out.append(f"The furthest-reaching relative with a recorded protocol used {label} and reached {LADDER[s]}.")

    n_cens = sum(r["censored"] for r in relatives)
    if n_cens:
        out.append(f"{n_cens} of the {len(relatives)} relatives found {'was' if n_cens == 1 else 'were'} "
                   "closed at centre shutdown rather than by any result, so "
                   f"{'it is' if n_cens == 1 else 'they are'} left out of these counts.")
    return out


def informative_first(r: dict, gate: int) -> tuple:
    """Display order: uncensored relatives that reached the bottleneck step, then those that
    got furthest, with identity breaking ties.

    Sorting by identity alone showed DnaK (P0A6Y8) six 80-100% identical relatives that
    were never even cloned, and said "none reached purified", while a 78% relative in the
    table below had reached crystallised. The closest sequence is not the most informative
    record: one that met the same wall is.
    """
    ms = r["max_stage"] if r["max_stage"] is not None else -1
    return (r["censored"], ms < gate, -ms, -(r["identity"] or 0))


def what_relatives_tried(con, precedents: list[dict], gate: int) -> dict:
    """The panel payload: facts about the bottleneck gate from EVERY retrieved relative,
    then the most informative relatives in detail."""
    trials = trials_for(con, [p["target_id"] for p in precedents])
    rel = []
    for p in precedents:
        ts = trials.get(p["target_id"], [])
        rel.append({"target_id": p["target_id"], "centre": p.get("centre", ""),
                    "organism": p.get("organism", ""), "identity": p.get("identity"),
                    "max_stage": p.get("max_stage"), "censored": bool(p.get("censored")),
                    "host": HOST_LABEL.get(p.get("host") or "", p.get("host") or ""),
                    "tag": TAG_LABEL.get(p.get("tag") or "", p.get("tag") or ""),
                    "n_trials": len(ts), "trials": ts})
    # facts from every relative and every trial; only the display is trimmed
    found = facts(rel, gate) if rel else []
    shown = sorted(rel, key=lambda r: informative_first(r, gate))[:MAX_RELATIVES]
    for r in shown:
        r["trials"] = r["trials"][:MAX_TRIALS]
    return {"gate": gate, "facts": found, "n_relatives": len(rel), "relatives": shown}
