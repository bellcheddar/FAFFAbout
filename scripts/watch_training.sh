#!/usr/bin/env bash
# watch_training.sh: supervise a long MLX run, and report the things that actually go wrong.
#
# Round 01 taught this script its job. Over one night the run was threatened three times,
# and the supervisor of the day would have caught exactly none of them:
#
#   1. DIVERGENCE. At lr 1e-4 the loss went 0.234 -> 4.831 -> 13.217 in twenty iterations.
#      A diverging run is healthy by every process-level measure: alive, consuming GPU,
#      writing its log on schedule. Watching the process would have watched the adapter
#      destroy itself for seven hours.
#   2. STARVATION. macOS indexing held the machine at load 42 on 10 cores and the run fell
#      to 29.4 s/iter against a REPORTED 14.7 s/iter. Nothing errored, the loss kept
#      falling, the log kept arriving. Only iterations counted against the wall clock
#      showed it, which is why this script times its own polls rather than trusting the
#      trainer's It/sec: that figure excludes the time the process is descheduled for.
#   3. DISK. The volume drifts down while checkpoints are written.
#
# Everything here reports on TRANSITION, not on every poll, because each line becomes a
# notification. A supervisor that cries every ninety seconds gets ignored, which is a
# slower way of not having one.
#
# The divergence test is gated on iter > 50: early training legitimately spikes, and an
# ungated version fired on iteration 10 of a run that was perfectly fine.
#
# Usage
#   bash scripts/watch_training.sh [log] [poll_seconds]
#   bash scripts/watch_training.sh data/train_round01.log 90
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG=${1:-data/train_round01.log}
INTERVAL=${2:-90}
DISK_MIN_GB=${DISK_MIN_GB:-5}
CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)
# Override with STARVE_LOAD, and set it absurdly high to silence the check entirely.
# Once starvation is DIAGNOSED and its remedy is known but out of reach (it needed root
# during round 01), repeating the alert every few minutes only wakes the supervisor's
# reader and takes CPU from the very job being starved. An alert you cannot act on is noise.
STARVE_AT=${STARVE_LOAD:-$(echo "$CORES * 2" | bc)}

[[ -f "$LOG" ]] || { echo "no such log: $LOG"; exit 2; }

# Seed from the log so a re-arm does not replay history as new events.
iter_now() { tr '\r' '\n' < "$LOG" | awk '/^Iter [0-9]+:/ {it=$2; sub(":","",it); if (it+0>m) m=it+0} END{print m+0}'; }
errs_now() { tr '\r' '\n' < "$LOG" | grep -cE 'Traceback|Error|error:|FAILED|Killed|OOM|out of memory|RuntimeError'; }

LAST=$(iter_now); PERRS=$(errs_now); T_LAST=$(date +%s)
STARVED=0; LOWDISK=0
echo "watching $LOG from iter $LAST (poll ${INTERVAL}s, starve above load $STARVE_AT on $CORES cores)"

while true; do
  sleep "$INTERVAL"
  NOW=$(date +%s)

  # --- the log: divergence, validation, milestones, completion ------------------------
  OUT=$(tr '\r' '\n' < "$LOG" | awk -v last="$LAST" '
    /^Iter [0-9]+: Train loss/ {
      it=$2; sub(":","",it); it+=0; l=$5; sub(",","",l); l+=0
      if (it > last && it > 50 && mn > 0 && l > 4*mn)
        printf "DIVERGENCE: iter %d loss %.3f is %.1fx the running min %.3f\n", it, l, l/mn, mn
      if (it > last && it % 500 == 0) printf "iter %d: train loss %.3f (best %.3f)\n", it, l, (l<mn?l:mn)
      if (mn == 0 || l < mn) mn = l
      if (it > mx) mx = it
    }
    /^Iter [0-9]+: Val loss/ {
      it=$2; sub(":","",it); it+=0; v=$5; sub(",","",v)
      if (it > last) printf "iter %d: VAL loss %s (best train %.3f)\n", it, v, mn
      if (it > mx) mx = it
    }
    /Saved final weights/ { done = $0 }
    END { if (done) print "TRAINING COMPLETE: " done; print "__MAX__" mx }
  ')
  NEW=$(printf '%s\n' "$OUT" | sed -n 's/^__MAX__//p' | tail -1)
  printf '%s\n' "$OUT" | grep -v '^__' | grep -v '^$'

  # --- real throughput, measured against the clock, not against the trainer's claim ---
  if [[ -n "${NEW:-}" && "$NEW" -gt "$LAST" ]]; then
    DI=$((NEW - LAST)); DT=$((NOW - T_LAST))
    if [[ "$DI" -gt 0 && "$DT" -gt 0 ]]; then
      SPI=$(echo "scale=1; $DT / $DI" | bc)
      # Only speak at milestones; the number is carried, not announced every poll.
      if [[ $((NEW / 500)) -gt $((LAST / 500)) ]]; then
        echo "  measured ${SPI}s/iter over the last ${DT}s (the trainer's own It/sec excludes descheduling)"
      fi
    fi
    LAST=$NEW; T_LAST=$NOW
  fi

  # --- new failure signatures ---------------------------------------------------------
  ERRS=$(errs_now)
  if [[ "${ERRS:-0}" -gt "${PERRS:-0}" ]]; then
    echo "FAILURE SIGNATURE in the log (count $PERRS -> $ERRS):"
    tr '\r' '\n' < "$LOG" | grep -E 'Traceback|Error|error:|FAILED|Killed|OOM|out of memory|RuntimeError' | tail -3
    PERRS=$ERRS
  fi

  # --- starvation, on transition only -------------------------------------------------
  LOAD=$(uptime | sed -n 's/.*load averages*: *\([0-9.]*\).*/\1/p')
  if [[ -n "${LOAD:-}" ]] && (( $(echo "$LOAD > $STARVE_AT" | bc -l) )); then
    if [[ "$STARVED" -eq 0 ]]; then
      STARVED=1
      echo "STARVED: load $LOAD on $CORES cores. Top three consumers:"
      ps -Ao pcpu,comm -r | sed -n '2,4p' | sed 's/^/    /'
      echo "    remedy: nohup ./scripts/spotlight_reaper.sh 40 >> reaper.log 2>&1 &"
    fi
  elif [[ "$STARVED" -eq 1 ]]; then
    STARVED=0; echo "recovered: load back to $LOAD on $CORES cores"
  fi

  # --- disk, on transition only -------------------------------------------------------
  FREE=$(df -g "$ROOT" | tail -1 | awk '{print $4}')
  if [[ "${FREE:-99}" -lt "$DISK_MIN_GB" ]]; then
    [[ "$LOWDISK" -eq 0 ]] && { LOWDISK=1; echo "LOW DISK: ${FREE} GB free, below the ${DISK_MIN_GB} GB floor"; }
  elif [[ "$LOWDISK" -eq 1 ]]; then
    LOWDISK=0; echo "disk recovered: ${FREE} GB free"
  fi

  # --- liveness, by resident memory and never by argv ---------------------------------
  # pgrep -f and ps | grep both match the shell running the check, because its own command
  # line contains the pattern. This produced several false readings before it was dropped.
  if [[ "$(ps -Ao pid,rss,command | awk '$2>1048576 && /mlx_lm lora/ {c++} END{print c+0}')" == "0" ]]; then
    echo "TRAINER GONE: no mlx_lm lora process holding more than 1 GB resident"
    tr '\r' '\n' < "$LOG" | tail -4
    exit 1
  fi

  # Completion ends the watch rather than leaving it armed over a finished run.
  tr '\r' '\n' < "$LOG" | grep -q "Saved final weights" && exit 0
done
