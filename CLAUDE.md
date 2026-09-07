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
