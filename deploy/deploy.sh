#!/usr/bin/env bash
# deploy.sh: ship FAFFAbout to the droplet.
#
# Nothing here runs without --go. The default is a dry run that prints exactly what would
# be transferred, because the payload is large and the box is shared with AlphaFraud.
#
# WHAT HAS TO SHIP, measured rather than guessed (2026-09-15):
#
#   data/faffabout.duckdb      159 MB   targets, censoring, labels, features, taxonomy
#   data/search/               1.7 GB   MMseqs2 index: 0.4 s per search, or 4.1 s without it
#   baseline/models/            30 MB   the boosters and their schema
#   app/                       312 KB
#                            ------
#                              1.9 GB
#
# Dropping data/search/archiveDB.idx* saves 1.7 GB of the 1.9 GB and costs 8 to 11x on
# every search. --no-index does that; decide deliberately rather than by default.
#
# The raw archive, the SFT corpus, the clusters and the adapters do NOT ship: they are
# build inputs, regenerated from Zenodo 10.5281/zenodo.821654.
#
# Usage
#   bash deploy/deploy.sh                 # dry run, prints the transfer
#   bash deploy/deploy.sh --go            # transfer and restart
#   bash deploy/deploy.sh --go --no-index # leave the 1.7 GB index behind
#   bash deploy/deploy.sh --code-only     # app + templates only, no data
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# .env supplies DROPLET (user@host) and REMOTE (absolute path). Never hard-code them.
[[ -f .env ]] && set -a && . ./.env && set +a
DROPLET="${FAFFABOUT_DROPLET:-${DROPLET:-}}"
REMOTE="${FAFFABOUT_REMOTE:-/srv/faffabout}"
SERVICE="${FAFFABOUT_SERVICE:-faffabout}"

GO=0; NO_INDEX=0; CODE_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --go) GO=1; shift ;;
    --no-index) NO_INDEX=1; shift ;;
    --code-only) CODE_ONLY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$DROPLET" ]]; then
  echo "No droplet configured. Put this in .env (which is gitignored):" >&2
  echo "  FAFFABOUT_DROPLET=root@203.0.113.10" >&2
  echo "  FAFFABOUT_REMOTE=/srv/faffabout" >&2
  exit 1
fi

# --- preflight on what we are about to send ------------------------------------------
for required in data/faffabout.duckdb baseline/models/gate_0.txt \
                baseline/models/schema.json app/server.py app/templates/rig.html; do
  [[ -e "$required" ]] || { echo "missing $required: build it before deploying" >&2; exit 1; }
done
# A booster without its schema silently predicts about the wrong centre, so refuse to ship
# a half set.
for p in "" declared_; do
  if [[ -e "baseline/models/${p}gate_0.txt" && ! -e "baseline/models/${p}schema.json" ]]; then
    echo "baseline/models/${p}gate_0.txt has no ${p}schema.json; re-run gbm_baseline.py" >&2
    exit 1
  fi
done

RSYNC=(rsync -az --human-readable --info=stats1,progress2)
[[ "$GO" -eq 1 ]] || RSYNC+=(--dry-run)

# NOTE: no --delete on the data tree. A deploy rsync that deletes has previously wiped
# live state on another project; anything the server writes stays out of the deploy path.
CODE_EXCLUDES=(--exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store')

echo "target      : $DROPLET:$REMOTE"
echo "mode        : $([[ $GO -eq 1 ]] && echo TRANSFER || echo 'DRY RUN (pass --go)')"
echo "search index: $([[ $NO_INDEX -eq 1 ]] && echo 'excluded (4.1 s per search)' || echo 'included (0.4 s per search, 1.7 GB)')"
echo

echo "--- code ---"
"${RSYNC[@]}" "${CODE_EXCLUDES[@]}" app/ "$DROPLET:$REMOTE/app/"
"${RSYNC[@]}" "${CODE_EXCLUDES[@]}" scripts/features_seq.py scripts/taxonomy.py "$DROPLET:$REMOTE/scripts/"
"${RSYNC[@]}" requirements.txt "$DROPLET:$REMOTE/"
"${RSYNC[@]}" "${CODE_EXCLUDES[@]}" baseline/models/ "$DROPLET:$REMOTE/baseline/models/"

if [[ "$CODE_ONLY" -eq 0 ]]; then
  echo "--- data ---"
  "${RSYNC[@]}" data/faffabout.duckdb "$DROPLET:$REMOTE/data/"
  "${RSYNC[@]}" data/parquet/taxonomy_lookup.parquet "$DROPLET:$REMOTE/data/parquet/"
  DATA_EX=()
  [[ "$NO_INDEX" -eq 1 ]] && DATA_EX+=(--exclude 'archiveDB.idx*')
  "${RSYNC[@]}" "${DATA_EX[@]}" data/search/ "$DROPLET:$REMOTE/data/search/"
fi

if [[ "$GO" -eq 1 ]]; then
  echo
  echo "--- restart and verify ---"
  ssh "$DROPLET" "systemctl restart $SERVICE && sleep 4 && systemctl is-active $SERVICE"
  # Verify by fetching the live page, not by trusting the restart.
  HOST="${FAFFABOUT_HOST:-faffabout.mdeller.com}"
  code=$(curl -s -o /dev/null -w '%{http_code}' "https://$HOST/healthz" || true)
  echo "https://$HOST/healthz -> $code"
  curl -s "https://$HOST/healthz" | head -c 400; echo
  [[ "$code" == "200" ]] || { echo "healthz is not 200: check 'journalctl -u $SERVICE -n 50'" >&2; exit 1; }
else
  echo
  echo "Dry run only. Re-run with --go to transfer."
fi
