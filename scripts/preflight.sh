#!/usr/bin/env bash
# preflight.sh: refuse to start a training run that is going to waste a night.
#
# Run this before EVERY launch. The checks are ordered by how much they have cost before:
# swap during a rank-64 run is fatal, Spotlight's daemon family recurringly starves
# training CPU for hours, and a machine up for a week runs at a fraction of its speed.
#
# Usage: bash scripts/preflight.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAIL=0
warn() { printf "  \033[33mWARN\033[0m  %s\n" "$*"; }
bad()  { printf "  \033[31mFAIL\033[0m  %s\n" "$*"; FAIL=1; }
ok()   { printf "  \033[32m ok \033[0m  %s\n" "$*"; }

echo "=== FAFFAbout preflight ==="

# --- disk ---------------------------------------------------------------------------
FREE_GB=$(df -g "$ROOT" | tail -1 | awk '{print $4}')
NEED_GB=14   # 8.54 GB model + adapters + checkpoints + headroom
if   [[ "$FREE_GB" -lt "$NEED_GB" ]]; then bad "only ${FREE_GB} GB free, need about ${NEED_GB} GB (model is 8.54 GB)"
elif [[ "$FREE_GB" -lt 25 ]];        then warn "${FREE_GB} GB free: enough to train, not enough to fuse with --de-quantize (~16 GB more)"
else ok "${FREE_GB} GB free"; fi

# --- swap ---------------------------------------------------------------------------
SWAP=$(sysctl -n vm.swapusage 2>/dev/null | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p')
if [[ -n "${SWAP:-}" ]] && (( $(echo "${SWAP:-0} > 100" | bc -l) )); then
  bad "swap in use (${SWAP}M): reboot before training"
else ok "swap clear"; fi

# --- uptime -------------------------------------------------------------------------
UP_DAYS=$(uptime | sed -n 's/.* up  *\([0-9]*\) day.*/\1/p')
if [[ -n "${UP_DAYS:-}" && "${UP_DAYS}" -ge 2 ]]; then
  warn "up for ${UP_DAYS} days: a reboot has recovered 2.3x throughput before"
else ok "uptime fine"; fi

# --- staged macOS update -------------------------------------------------------------
if [[ -n "$(ls -A /Library/Updates 2>/dev/null | grep -v '^\.')" ]]; then
  warn "a staged macOS update in /Library/Updates generates background work until installed"
else ok "no staged macOS update"; fi

# --- Spotlight ----------------------------------------------------------------------
# sudo has no TTY in an agent session, so this can only report, never fix.
IDX=$(mdutil -a -s 2>/dev/null | grep -c "Indexing enabled" || true)
if [[ "${IDX:-0}" -gt 0 ]]; then
  warn "Spotlight indexing is ENABLED on ${IDX} volume(s). In a real Terminal: sudo mdutil -a -i off"
else ok "Spotlight indexing off"; fi

BUSY=$(ps -Ao pcpu,comm | awk '$1 > 30 && ($2 ~ /mds|mdworker|MediaAnalysis|photoanalysis|CoreSpotlight|mobileassetd/)' | head -5)
if [[ -n "$BUSY" ]]; then
  warn "background analysers above 30% CPU:"; echo "$BUSY" | sed 's/^/          /'
  warn "start the reaper: nohup ./scripts/spotlight_reaper.sh 40 >> reaper.log 2>&1 &"
else ok "no background analyser above 30% CPU"; fi

# --- data ---------------------------------------------------------------------------
for f in data/sft/train.jsonl data/sft/valid.jsonl config/train_config.yaml; do
  [[ -s "$ROOT/$f" ]] && ok "$f present" || bad "$f missing or empty"
done

# iCloud eviction: a dataless file reads as present and stalls the run on first access.
EVICTED=$(find "$ROOT/data/sft" -type f -exec sh -c 'test $(stat -f %z "$1") -gt 0 || echo "$1"' _ {} \; 2>/dev/null | head -3)
[[ -n "$EVICTED" ]] && bad "zero-length (possibly evicted) files: $EVICTED" || ok "no evicted data files"

# --- mlx ----------------------------------------------------------------------------
if "$ROOT/.venv/bin/python" -c "import mlx_lm" 2>/dev/null; then
  V=$("$ROOT/.venv/bin/python" -c "import mlx_lm,sys; sys.stdout.write(getattr(mlx_lm,'__version__','?'))")
  ok "mlx_lm $V installed"
else bad "mlx_lm not installed: uv pip install mlx-lm"; fi

echo
if [[ "$FAIL" -eq 1 ]]; then
  printf "\033[31mPREFLIGHT FAILED: do not launch.\033[0m\n"; exit 1
fi
printf "\033[32mPreflight passed.\033[0m Warnings above are worth clearing first.\n"
