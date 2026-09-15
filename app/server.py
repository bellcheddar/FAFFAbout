"""server.py: the Flask app.

  GET  /                 the Pipeline Rig
  POST /api/resolve      whatever was pasted -> a sequence, shown before anything is computed
  POST /api/predict      the full forecast (spec section 7.3)
  GET  /api/archive      the archive-map summary
  GET  /healthz          readiness, including which pieces are actually present

Run locally:
  .venv/bin/python -m flask --app app/server.py run --port 8009 --debug

In production, gunicorn behind nginx on faffabout.mdeller.com. The DuckDB file and the
Parquet directory are opened READ-ONLY, so several workers can share them.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "scripts"))

import predict as P          # noqa: E402
import resolve as R          # noqa: E402
import retrieve as RET       # noqa: E402

DB = ROOT / "data" / "faffabout.duckdb"

app = Flask(__name__, template_folder=str(ROOT / "app" / "templates"),
            static_folder=str(ROOT / "app" / "static"))
app.config["JSON_SORT_KEYS"] = False

_con: duckdb.DuckDBPyConnection | None = None


def con() -> duckdb.DuckDBPyConnection:
    """One read-only connection per process, opened lazily."""
    global _con
    if _con is None:
        _con = duckdb.connect(str(DB), read_only=True)
    return _con


def choices_from(payload: dict) -> P.Choices:
    return P.Choices(
        method=payload.get("method", "xray"),
        centre=payload.get("centre", "JCSG"),
        host=payload.get("host", "") or "",
        tag=payload.get("tag", "") or "",
        protease=payload.get("protease", "") or "",
        codon_optimised=bool(payload.get("codon_optimised")),
        autoinduction=bool(payload.get("autoinduction")),
    )


@app.get("/")
def index():
    return render_template("rig.html", stats=archive_stats())


@app.post("/api/resolve")
def api_resolve():
    text = (request.json or {}).get("query", "")
    try:
        res = R.resolve(text)
    except R.ResolveError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"lookup failed: {type(e).__name__}"}), 502
    return jsonify(res.as_dict())


@app.post("/api/predict")
def api_predict():
    payload = request.json or {}
    t0 = time.time()
    try:
        res = R.resolve(payload.get("query", ""))
    except R.ResolveError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"lookup failed: {type(e).__name__}"}), 502

    try:
        precs = RET.precedents(res.sequence, con=con())
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"retrieval failed: {type(e).__name__}"}), 500

    ctx = RET.context(precs)
    out = P.forecast(res, ctx, precs, choices_from(payload))

    # The narrative is the ONLY field a language model may write, and it is optional.
    try:
        import llm
        narrative = llm.narrate(out)
        if narrative:
            out["model"]["narrative"] = narrative
            out["model"]["narrative_from"] = llm.MODEL_NAME
    except Exception:  # noqa: BLE001
        out["caveats"].append("No fine-tuned model is serving, so this forecast has "
                              "numbers and precedents but no written interpretation.")

    out["timing_ms"] = int((time.time() - t0) * 1000)
    return jsonify(out)


def archive_stats() -> dict:
    try:
        n_t, n_c, n_dep = con().execute("""
            SELECT (SELECT count(*) FROM targets),
                   (SELECT count(*) FROM censoring WHERE censored),
                   (SELECT count(*) FROM censoring WHERE max_stage >= 8)
        """).fetchone()
        return {"targets": n_t, "censored": n_c, "deposited": n_dep,
                "censored_pct": round(100.0 * n_c / n_t, 1) if n_t else 0.0}
    except Exception:  # noqa: BLE001
        return {"targets": 0, "censored": 0, "deposited": 0, "censored_pct": 0.0}


@app.get("/api/archive")
def api_archive():
    """Outcome counts per centre, for the archive map."""
    rows = con().execute("""
        SELECT c.centre, count(*) AS n,
               count(*) FILTER (WHERE c.max_stage >= 8) AS deposited,
               count(*) FILTER (WHERE c.censored) AS censored,
               round(100.0 * avg(CASE WHEN c.max_stage >= 8 THEN 1 ELSE 0 END), 2) AS pct_deposited
        FROM censoring c GROUP BY 1 ORDER BY n DESC
    """).df().to_dict("records")
    return jsonify({"centres": rows, "stats": archive_stats()})


@app.get("/healthz")
def healthz():
    """Report what is actually present, so a half-provisioned box is obvious."""
    checks = {
        "duckdb": DB.exists(),
        "search_index": (ROOT / "data" / "search" / "archiveDB.idx").exists(),
        "boosters_headline": (ROOT / "baseline" / "models" / "gate_0.txt").exists(),
        "boosters_declared": (ROOT / "baseline" / "models" / "declared_gate_0.txt").exists(),
        "taxonomy": (ROOT / "data" / "external" / "nodes.dmp").exists(),
    }
    try:
        import llm
        checks["llm"] = bool(llm.available())
    except Exception:  # noqa: BLE001
        checks["llm"] = False
    ok = all(v for k, v in checks.items() if k != "llm")
    return jsonify({"ok": ok, "checks": checks, "stats": archive_stats()}), (200 if ok else 503)


if __name__ == "__main__":
    app.run(port=8009, debug=True)
