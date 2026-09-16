#!/usr/bin/env bash
# Apple's background metadata/media analysers compete with a long GPU run and launchd respawns
# each one seconds after a kill. This holds their duty cycle down for the length of a run rather
# than stopping them; the durable fix is `sudo mdutil -a -i off` plus pausing Photos analysis, in
# a real Terminal, because sudo has no TTY here.
#
# Deliberately narrow. Only background *analysers*, only above a CPU threshold, and never
# fileproviderd -- signalling that freezes every I/O under Documents (iCloud) and hangs the shell.
THRESH=${1:-40}
# mobileassetd downloads OS assets and models. Safe to signal -- it retries, and the download
# resuming tomorrow is a far better outcome than it competing with an overnight training run.
# spotlightknowledged(.updater) and hybridsearchd were added after round 01: the machine hit
# load 42 with spotlightknowledged.updater at 85% CPU, and NONE of the names below matched it,
# so the reaper and preflight both reported all clear while the run lost 40% of its wall clock.
# Same family as mds_stores and mdworker, same respawn behaviour.
#
# Deliberately still NOT here: ARDAgent's build_hd_index, which sat at 48% during the same
# window. It belongs to Remote Management rather than to Spotlight, so killing it is Marc's
# call, not a default. If a run is being starved and this reaper is not enough, that is the
# next thing to look at.
PATTERN='Metadata\.framework|CoreSpotlight|spotlightknowledged|hybridsearchd|mds_stores|mdworker|MediaAnalysis|photoanalysisd|photolibraryd|AppleNeuralEngine|suggestd|knowledge-agent|mobileassetd'
while true; do
  ps -eo pid,pcpu,args | \
    awk -v t="$THRESH" -v p="$PATTERN" \
        '$2+0 > t && $0 ~ p && $0 !~ /fileproviderd/ && $0 !~ /awk/ {print $1, $2}' | \
  while read -r pid cpu; do
    kill "$pid" 2>/dev/null && printf '%s | reaped %s at %s%% cpu\n' "$(date '+%H:%M:%S')" "$pid" "$cpu"
  done
  # 20s let a daemon at 195% CPU run for up to twenty seconds before it was caught; over an
  # eight-hour run that sawtooth is most of the damage. `ps` costs nothing next to that.
  sleep 5
done
