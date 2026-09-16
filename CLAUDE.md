# FAFFAbout: working rules for Claude Code

Spec: `faffabout_build_spec_v1.md` (source of truth). Live status: `PROJECT_PLAN.md`.

## Hard rules

1. **Keep going through the phases.** When a phase's acceptance test passes: update
   PROJECT_PLAN.md, commit, and start the next phase in the same turn. Pause only for a
   decision that is genuinely Marc's: a scientific default or threshold, a licence, anything
   destructive or outward-facing, anything needing `sudo`.
2. **Censored is never a negative.** Any code that builds labels excludes censored rows from
   the loss, keeps them in retrieval context, and surfaces the flag in output.
3. **Splits are by MMseqs2 cluster (30% identity), never by row.**
4. **The raw sequence never goes in a prompt.** Features only.
5. **The LLM writes the narrative only.** Every number comes from the GBM or DuckDB. A model
   that emits a target ID, PDB ID or probability is a bug.
6. **Never commit `data/`, `adapters/`, `models/`.** They regenerate from Zenodo
   10.5281/zenodo.821654 (CC-BY-SA-4.0). Stage files explicitly; never `git add -A`.
7. **British English, no em dashes** (colons or parentheses).
8. **Front end uses the Pipeline Rig palette** in `faffabout_rig_v1.html`, not the
   marcdeller.com header.
9. **Run `scripts/preflight.sh` before any training launch** and supervise every run over
   30 minutes with a Monitor.

## Environment

- Python 3.14 in `.venv` (uv). Run scripts as `.venv/bin/python scripts/NN_name.py`.
- `mmseqs` on PATH via Homebrew.
- `ugrep` is aliased as `grep`: it treats the Latin-1 XML as binary and returns nothing.
  Use `awk` or `grep -a` on archive files.
- The tarball ships AppleDouble `._*.xml.gz` forks; every glob over `TargetsbyContributor/`
  must exclude names starting with `._`.
- **`timeout` and `gtimeout` do not exist here** (BSD userland, no coreutils). Wrapping a
  command in `timeout` returns 127 and the command never runs, which silently voids
  whatever the check was meant to prove.
- **Any detached/background job started from here lands at nice 5, whatever the parent.**
  Measured, after two wrong guesses: a foreground tool call is nice 0, but a `nohup ... &`
  child spawned *from that same nice-0 shell* comes back nice 5. So it is not inherited from
  the session and it is not a foreground/background property of the tool call: detaching is
  what demotes it. Confirmed across the round 01 trainer, its `08_train.py` parent,
  `wandb-core`, and `spotlight_reaper.sh`, all at nice 5, with our code never asking for it.
  A nice-5 process cannot outrank ordinary nice-0 work: the run fell to 0.9% CPU and
  204 s/iter while the machine sat at load 13, an ETA of 3.8 days. It also explains a
  `sleep 60` in a background task taking sixteen minutes.
  **There is no way to launch a long job from here that avoids this**, and it cannot be
  undone afterwards: `renice` needs root to LOWER a value, `sudo` has no TTY, and
  `taskpolicy -B` reports success while changing nothing. The only repair is Marc's, in a
  real Terminal: `sudo renice -n 0 -p <pid>` (or `sudo taskpolicy -B -p <pid>`), which
  fixes a RUNNING job in place with nothing lost.
  The mechanism is the throttled QoS band, not the nice number alone: the threads show
  PRI `31T`, and on Apple Silicon that band is confined to the two efficiency cores and
  kept off the eight performance cores. That is why the load average is useless here, and
  why it looks so wrong: the machine sat at load 8.57 on 10 cores with P-cores free while
  the trainer got 2.5%. **GPU contention was tested and ruled out** (`ioreg -c
  AGXAcceleratorG13X` showed Device Utilization 36%, so the GPU was idle waiting for work
  the trainer could not submit, rather than being fought over). `08_train.py` now hard-stops on nonzero nice (`--allow-low-priority` to
  override), verified under `nice -n 5`. **Check `ps -o nice` early on any run that is
  slow for no visible reason**: every other signal will look healthy.
- **An empty filter result is not evidence of absence.** A process check printed "nothing
  above 1%" directly beneath an 84.9% `spotlightknowledged` process, because the pattern
  said `Spotlight` and the process is lower case. The same hand-maintained name list is
  duplicated in `preflight.sh` and `spotlight_reaper.sh`, so one stale name blinds all
  three at once. Match with `tolower()`, and **always print the top consumers unfiltered
  next to the filtered answer** so a missed name is visible rather than silent.
- **A starved run looks perfectly healthy.** Round 01 ran at 29 s/iter against a reported
  14.7 s/iter while macOS indexing held the machine at load 42 on 10 cores. The process was
  alive, the loss was falling, the log was on schedule and nothing errored: the only signal
  was iterations counted against the wall clock. Check progress against `date`, not against
  the trainer's own It/sec, which excludes the time it is being descheduled for.
- **Never establish process liveness by matching argv.** `pgrep -f X` and `ps | grep X`
  both match the shell running the check, because its own command line contains X, and the
  `[x]` bracket trick only protects grep from itself, not the parent zsh. This produced
  five separate false readings in one session, including an apparent duplicate training
  job. Count by resident memory instead (`ps -Ao pid,rss,command | awk '$2 > 1048576 ...'`),
  or read a background job's output file. Also note `awk '/pattern/'` exits 0 whether or
  not it matched, so `awk ... && echo "found"` always fires.
