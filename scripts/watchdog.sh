#!/usr/bin/env bash
# Generic stall supervisor for long unattended runs. Source it, or copy `supervise` out.
#
# Every line it prints on stdout is an event: pipe it through Monitor so stalls and restarts
# arrive as notifications instead of being discovered hours later.
#
#   supervise NAME PROGRESS_CMD PATTERN START_CMD STALL_SECONDS DONE_CMD
#
#   NAME          label used in events
#   PROGRESS_CMD  echoes a value that CHANGES while healthy (a line count, a file count)
#   PATTERN       pgrep -f pattern identifying the process
#   START_CMD     relaunches it; must be resumable, i.e. pick up where it left off
#   STALL_SECONDS no change for this long => restart
#   DONE_CMD      succeeds when the stage is finished -- and MUST NOT be satisfiable mid-run.
#                 `[ -f adapters.safetensors ]` looks right and is wrong: mlx_lm writes that file
#                 at the FIRST checkpoint, so it fired at iteration 100 of 8,000, the supervisor
#                 declared victory and exited, and an 8-hour job then ran unwatched. Use a signal
#                 that only the END produces -- the final iteration appearing in the log.
emit() { printf '%s | %s\n' "$(date '+%H:%M:%S')" "$*"; }

supervise() {
  local name=$1 progress=$2 pattern=$3 start=$4 stall=$5 done_cmd=$6
  local last="" changed beat restarts=0 now cur
  changed=$(date +%s); beat=$changed
  while true; do
    if eval "$done_cmd" 2>/dev/null; then emit "$name: COMPLETE"; return 0; fi
    now=$(date +%s)

    if ! pgrep -f "$pattern" >/dev/null 2>&1; then
      restarts=$((restarts + 1))
      if [ "$restarts" -gt 5 ]; then emit "$name: 5 restarts exhausted, giving up"; return 1; fi
      emit "$name: process gone, restarting (attempt $restarts)"
      eval "$start"; sleep 90; changed=$(date +%s); last=""; continue
    fi

    cur=$(eval "$progress" 2>/dev/null | tr -d ' ')
    if [ -n "$cur" ] && [ "$cur" != "$last" ]; then
      last=$cur; changed=$now
    elif [ $((now - changed)) -gt "$stall" ]; then
      restarts=$((restarts + 1))
      if [ "$restarts" -gt 5 ]; then emit "$name: 5 restarts exhausted, giving up"; return 1; fi
      emit "$name: STALLED at '$cur' for $(((now - changed) / 60))m, restarting (attempt $restarts)"
      pkill -f "$pattern" 2>/dev/null; sleep 25
      pgrep -f "$pattern" >/dev/null && { pkill -9 -f "$pattern"; sleep 10; }
      eval "$start"; sleep 90; changed=$(date +%s); last=""
    fi

    # Heartbeat, so silence means dead rather than merely quiet.
    if [ $((now - beat)) -ge 1800 ]; then emit "$name: alive, progress '$cur'"; beat=$now; fi
    sleep 60
  done
}
