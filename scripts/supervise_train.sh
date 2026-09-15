#!/usr/bin/env bash
# supervise_train.sh: watch a training run and emit one line per event.
#
# Pipe through Monitor so stalls and failures arrive as notifications rather than being
# found hours later. A 6,000-iteration run is 6 to 10 hours on this Mac; an unwatched one
# that dies at iteration 300 wastes the night.
#
# Progress signal: the highest iteration number in the log. It changes while healthy,
# unlike CPU% (which reads near zero for a GPU-bound MLX process that is working fine) or
# process liveness (which stays true throughout a stall).
#
# Done condition: the final-weights line, which only the END produces. NOT the existence of
# adapters.safetensors: mlx_lm writes that at the FIRST checkpoint, so it is true by
# iteration 500 of 6,000 and a supervisor keyed on it declares victory and exits.
#
# Usage: bash scripts/supervise_train.sh <log> [stall_seconds]
set -uo pipefail

LOG="${1:?usage: supervise_train.sh <log> [stall_seconds]}"
STALL="${2:-2400}"          # 40 min: an eval pass plus a checkpoint save is minutes, not seconds
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

emit() { printf '%s | %s\n' "$(date '+%H:%M:%S')" "$*"; }

last_iter() { grep -oE 'Iter [0-9]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+'; }

emit "supervising $LOG (stall window ${STALL}s)"
changed=$(date +%s); beat=$changed; last=""

while true; do
  now=$(date +%s)

  # --- terminal states, checked before anything else -------------------------------
  if grep -q "Saved final weights" "$LOG" 2>/dev/null; then
    emit "COMPLETE: $(grep -oE 'Iter [0-9]+.*Val loss [0-9.]+' "$LOG" | tail -1)"
    emit "final: $(grep 'Saved final weights' "$LOG" | tail -1)"
    exit 0
  fi
  # Widened deliberately: a filter that greps only for progress stays silent through a
  # crash, and silence is indistinguishable from "still running".
  if grep -qE 'Traceback|CUDA|Metal error|RuntimeError|ValueError|KeyError|out of memory|Killed|zsh: killed|Segmentation' "$LOG" 2>/dev/null; then
    emit "FAILED: $(grep -nE 'Traceback|RuntimeError|ValueError|KeyError|out of memory|Killed|Segmentation' "$LOG" | tail -1 | cut -c1-200)"
    emit "last progress: iter $(last_iter)"
    exit 1
  fi

  cur=$(last_iter)
  if [ -n "$cur" ] && [ "$cur" != "$last" ]; then
    # report every validation point, which is the number worth seeing
    v=$(grep -oE "Iter $cur: Val loss [0-9.]+" "$LOG" 2>/dev/null | tail -1)
    [ -n "$v" ] && emit "$v"
    last="$cur"; changed=$now
  elif [ $((now - changed)) -gt "$STALL" ]; then
    emit "STALLED at iter ${cur:-none} for $(((now - changed) / 60))m; check Activity Monitor and swap"
    changed=$now
  fi

  if [ $((now - beat)) -ge 1800 ]; then
    t=$(grep -oE 'Iter [0-9]+: Train loss [0-9.]+.*It/sec [0-9.]+' "$LOG" 2>/dev/null | tail -1)
    emit "alive at iter ${cur:-none}  ${t:-}"
    beat=$now
  fi
  sleep 60
done
