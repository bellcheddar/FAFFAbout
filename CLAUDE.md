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
- **Everything launched from the agent session inherits nice 5, including training.** The
  round 01 trainer, its `08_train.py` parent, `wandb-core` and even a bare `sleep` all ran
  at nice 5; our code never asks for it. This is survivable on an idle machine and crippling
  under competition, because a nice-5 process cannot outrank ordinary nice-0 work: the run
  fell to 0.9% CPU and roughly 200 s/iter while the machine sat at load 13. It also explains
  shell oddities in this session, such as a `sleep 60` taking sixteen minutes. **`renice`
  cannot undo it** (lowering a nice value needs root, and `sudo` has no TTY here) and
  `taskpolicy -B` reports success while changing nothing. Relaunching does not help either,
  since the child inherits it again. The fix is Marc's, in a real Terminal:
  `sudo renice -n 0 -p <pid>` on the running job. **Check `ps -o nice` early on any run that
  looks slow for no reason**, and prefer launching long training from a real Terminal.
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
