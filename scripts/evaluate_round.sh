#!/usr/bin/env bash
# evaluate_round.sh: everything that has to happen once a training round finishes.
#
# Five steps that are easy to get wrong at eight in the morning: serve the adapter, score
# the probabilities, check for hallucinated identifiers, measure whether the prose is worth
# anything, and compare that against the corpus floor. This does them in order and prints
# one summary.
#
# The question it exists to answer is the specification's: "if you cannot state a number
# the LLM adds over the GBM, you are shipping decoration." The GBM's numbers are already
# known (cluster-held-out mean AUROC 0.839, terminal 0.866), so the only open questions are
# whether the narrative is grounded and whether it is anything more than a recited template.
#
# Usage
#   bash scripts/evaluate_round.sh                       # newest adapter with weights
#   bash scripts/evaluate_round.sh --adapter adapters/...-round01
#   bash scripts/evaluate_round.sh --n 40                # cases per generative eval
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY=.venv/bin/python
PORT=8080
ADAPTER=""
N=40

while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter) ADAPTER="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$ADAPTER" ]]; then
  ADAPTER=$(ls -dt adapters/*/ 2>/dev/null | while read -r d; do
              [[ -f "$d/adapters.safetensors" ]] && { echo "${d%/}"; break; }; done)
fi
[[ -n "$ADAPTER" ]] || { echo "No adapter with weights. Training writes adapters.safetensors at the first checkpoint." >&2; exit 1; }

MODEL=$($PY -c "import yaml;print(yaml.safe_load(open('config/train_config.yaml'))['model'])")
echo "=============================================================="
echo " adapter : $ADAPTER"
echo " model   : $MODEL"
$PY - "$ADAPTER" <<'PYEOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "adapter_config.json"
if p.exists():
    d = json.loads(p.read_text())
    print(f" config  : {d.get('num_layers')} layers, rank {d.get('lora_parameters',{}).get('rank')}, "
          f"lr {d.get('learning_rate')}, iters {d.get('iters')}")
PYEOF
echo "=============================================================="

# --- serve -----------------------------------------------------------------------------
echo
echo "--- starting mlx_lm.server on 127.0.0.1:$PORT ---"
$PY -m mlx_lm server --model "$MODEL" --adapter-path "$ADAPTER" \
    --host 127.0.0.1 --port "$PORT" > data/eval_server.log 2>&1 &
SERVER_PID=$!
cleanup() { kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; }
trap cleanup EXIT

for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 2
done
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "server never came up; last lines of data/eval_server.log:" >&2
  tail -20 data/eval_server.log >&2
  exit 1
fi
# The model field must match the path the server resolved, or requests 404 with an opaque
# hub-lookup error that reads like a network fault.
MID=$(curl -s "http://127.0.0.1:$PORT/v1/models" | $PY -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])")
echo "  serving as: $MID"
export FAFFABOUT_LLM="http://127.0.0.1:$PORT/v1"
export FAFFABOUT_LLM_MODEL="$MID"

# --- 1. calibration, and the GBM delta --------------------------------------------------
echo
echo "--- 1/3 calibration (Brier, ECE, AUROC, bottleneck top-1, GBM delta) ---"
$PY eval/eval_calibration.py --llm-endpoint "$FAFFABOUT_LLM" --llm-model "$MID" --n 300 2>&1 | tail -30

# --- 2. hallucinated identifiers: the automatic fail ------------------------------------
echo
echo "--- 2/3 generative (hallucinated identifiers are an AUTOMATIC FAIL) ---"
$PY eval/eval_generative.py --llm-endpoint "$FAFFABOUT_LLM" --llm-model "$MID" --n "$N" 2>&1 | tail -20
GEN_RC=${PIPESTATUS[0]}

# --- 3. is the prose worth anything? ----------------------------------------------------
echo
echo "--- 3/3 narrative value, against the corpus floor ---"
$PY eval/eval_narrative_value.py --llm-endpoint "$FAFFABOUT_LLM" --llm-model "$MID" --n 24 2>&1 | tail -16

# --- summary ----------------------------------------------------------------------------
echo
echo "=============================================================="
echo " SUMMARY"
echo "=============================================================="
$PY - <<'PYEOF'
import json, pathlib
root = pathlib.Path(".")
nv = root / "eval" / "narrative_value.json"
cal = root / "eval" / "calibration.json"
if cal.exists():
    d = json.loads(cal.read_text())
    if "llm" in d:
        l = d["llm"]
        print(f"  LLM   Brier {l['brier']:.4f}  ECE {l['ece']:.4f}  AUROC {l.get('auroc') or float('nan'):.4f}")
    g = d.get("cluster", {}).get("gbm", {}).get("pooled")
    if g:
        print(f"  GBM   Brier {g['brier']:.4f}  ECE {g['ece']:.4f}  AUROC {g['auroc']:.4f}")
        if "llm" in d:
            dl = d["llm"]["brier"] - g["brier"]
            print(f"  delta Brier {dl:+.4f} -> " +
                  ("the LLM is better calibrated" if dl < 0 else
                   "the GBM is better calibrated, as expected; the LLM must earn its place on the prose"))
if nv.exists():
    v = json.loads(nv.read_text())
    print()
    print(f"  narrative: template echo {v['template_echo_mean']:.3f} (corpus floor 0.554)")
    print(f"             responsiveness {v['responsiveness_mean_pairwise']:.3f} (corpus floor 0.138)")
    b = v.get("bottleneck_top1")
    print(f"             bottleneck top-1 {b:.2f}" if b is not None else "             bottleneck top-1 n/a")
    if v["template_echo_mean"] > 0.554:
        print("  -> ECHO ABOVE THE FLOOR: the model recites its training completions more")
        print("     closely than the corpus resembles itself. That is decoration, not reasoning.")
    if v["responsiveness_mean_pairwise"] > 0.30:
        print("  -> narratives for different targets resemble each other: it is ignoring its input")
    for w in v.get("warnings", []):
        print(f"  WARNING: {w}")
PYEOF
echo
echo " hallucination check exited $GEN_RC (non-zero = invented identifiers = automatic fail)"
echo " grading form: eval/generative_review.md"
