#!/usr/bin/env bash
# serve_llm.sh: serve the fine-tuned adapter for the app's narrative field.
#
# The app calls an OpenAI-compatible endpoint at 127.0.0.1:8080 and asks it for prose
# only: every probability, count and interval in the payload is computed before this
# server is touched. If it is down, the forecast is still complete and simply has no
# written interpretation.
#
# THE TRAP. The "model" field in a request must match the path the server RESOLVED, not
# what you hoped it was called. A mismatch comes back as an opaque hub-lookup 404 that
# reads like a network fault. app/llm.py therefore asks /v1/models and uses whatever it
# reports; this script prints the same thing so the two can be compared by eye.
#
# Serving base + adapter avoids fusing. `mlx_lm.fuse --de-quantize` needs roughly 16 GB
# more disk and is only worth it for GGUF export or for handing the model to someone else.
#
# Usage
#   bash scripts/serve_llm.sh                       # newest adapter round
#   bash scripts/serve_llm.sh --adapter adapters/faffabout-llama-3.1-8b-8bit-round01
#   bash scripts/serve_llm.sh --port 8080 --check   # start, verify, then keep serving
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PORT=8080
ADAPTER=""
CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --adapter) ADAPTER="$2"; shift 2 ;;
    --check) CHECK=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

MODEL=$(.venv/bin/python -c "import yaml;print(yaml.safe_load(open('config/train_config.yaml'))['model'])")

# newest round that actually holds weights: a directory exists from the moment a run
# starts, but adapters.safetensors only appears at the first checkpoint
if [[ -z "$ADAPTER" ]]; then
  ADAPTER=$(ls -dt adapters/*/ 2>/dev/null | while read -r d; do
              [[ -f "$d/adapters.safetensors" ]] && { echo "${d%/}"; break; }; done || true)
fi
if [[ -z "$ADAPTER" ]]; then
  echo "No adapter with weights yet. A round directory appears when training starts, but" >&2
  echo "adapters.safetensors only lands at the first checkpoint (save_every in the config)." >&2
  exit 1
fi

echo "model   : $MODEL"
echo "adapter : $ADAPTER"
.venv/bin/python - "$ADAPTER" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "adapter_config.json"
if p.exists():
    d = json.loads(p.read_text())
    print(f"          {d.get('num_layers')} layers, rank "
          f"{d.get('lora_parameters', {}).get('rank')}, iters {d.get('iters')}")
PY
echo "port    : $PORT"
echo

.venv/bin/python -m mlx_lm server --model "$MODEL" --adapter-path "$ADAPTER" \
    --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# wait for it to answer rather than sleeping a guess
for _ in $(seq 1 90); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 2
done

echo
echo "=== what the server calls the model (use this verbatim as the \"model\" field) ==="
curl -s "http://127.0.0.1:$PORT/v1/models" | .venv/bin/python -c "
import json,sys
try:
    for m in json.load(sys.stdin).get('data', []):
        print('   ', m['id'])
except Exception as e:
    print('    could not read /v1/models:', e)
"
echo
echo "point the app at it with:  export FAFFABOUT_LLM=http://127.0.0.1:$PORT/v1"

if [[ "$CHECK" -eq 1 ]]; then
  echo
  echo "=== smoke test: one completion ==="
  MID=$(curl -s "http://127.0.0.1:$PORT/v1/models" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])")
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence: what usually stops a purified protein from crystallising?\"}],\"max_tokens\":80,\"temperature\":0.2}" \
    | .venv/bin/python -c "
import json,sys
d=json.load(sys.stdin)
print('   ', d['choices'][0]['message']['content'].strip()[:400])
"
fi

echo
echo "serving (ctrl-c to stop)"
wait $SERVER_PID
