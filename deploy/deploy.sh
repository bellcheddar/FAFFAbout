#!/usr/bin/env bash
# deploy.sh: ship FAFFAbout to the droplet.
#
# Nothing here runs without --go. The default is a dry run that prints exactly what would
# be transferred, because the payload is large and the box is shared with AlphaFraud.
#
# WHAT HAS TO SHIP, measured rather than guessed (2026-09-15):
#
#   data/faffabout_serving.duckdb  205 MB   real tables, built by scripts/build_serving_db.py
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
#   bash deploy/deploy.sh --go --no-restart   # FIRST deploy: ship, then run deploy/provision.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# .env supplies DROPLET (user@host) and REMOTE (absolute path). Never hard-code them.
[[ -f .env ]] && set -a && . ./.env && set +a
DROPLET="${FAFFABOUT_DROPLET:-${DROPLET:-}}"
REMOTE="${FAFFABOUT_REMOTE:-/opt/faffabout}"      # the droplet keeps every app in /opt/<app>
SERVICE="${FAFFABOUT_SERVICE:-faffabout-web}"
# FAFFABOUT_SSH_KEY names a deploy-only key, so nothing depends on ~/.ssh/config.
# One shared connection for every step: the droplet rate-limits new SSH connections, and a
# deploy that opened eight in a row was locked out mid-transfer on 2026-09-28.
SSH=(ssh -o ControlMaster=auto -o "ControlPath=/tmp/faffabout-ssh-%C" -o ControlPersist=300)
[[ -n "${FAFFABOUT_SSH_KEY:-}" ]] && SSH+=(-i "${FAFFABOUT_SSH_KEY/#\~/$HOME}" -o IdentitiesOnly=yes)

GO=0; NO_INDEX=0; CODE_ONLY=0; NO_RESTART=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --go) GO=1; shift ;;
    --no-index) NO_INDEX=1; shift ;;
    --code-only) CODE_ONLY=1; shift ;;
    --no-restart) NO_RESTART=1; shift ;;   # first deploy: ship, then run provision.sh
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$DROPLET" ]]; then
  echo "No droplet configured. Put this in .env (which is gitignored):" >&2
  echo "  FAFFABOUT_DROPLET=root@203.0.113.10" >&2
  echo "  FAFFABOUT_REMOTE=/opt/faffabout" >&2
  echo "  FAFFABOUT_SSH_KEY=~/.ssh/faffabout_deploy" >&2
  exit 1
fi

# --- preflight on what we are about to send ------------------------------------------
for required in data/faffabout_serving.duckdb baseline/models/gate_0.txt \
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

# macOS ships openrsync (2.6.9-compatible), which has no --info; Homebrew's rsync 3.x does,
# and --partial lets an interrupted 1.7 GB index transfer resume instead of restarting.
RSYNC_BIN=$(command -v /opt/homebrew/bin/rsync || command -v rsync)
RSYNC=("$RSYNC_BIN" -az --partial --human-readable --info=stats1,progress2 -e "${SSH[*]}")
[[ "$GO" -eq 1 ]] || RSYNC+=(--dry-run)

# NOTE: no --delete on the data tree. A deploy rsync that deletes has previously wiped
# live state on another project; anything the server writes stays out of the deploy path.
CODE_EXCLUDES=(--exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store')

echo "target      : $DROPLET:$REMOTE"
echo "mode        : $([[ $GO -eq 1 ]] && echo TRANSFER || echo 'DRY RUN (pass --go)')"
echo "search index: $([[ $NO_INDEX -eq 1 ]] && echo 'excluded (4.1 s per search)' || echo 'included (0.4 s per search, 1.7 GB)')"
echo

# rsync creates the final directory but not missing parents. An if-block, not `[[ ]] &&`:
# a failure on the right of && does not trip `set -e`, and an earlier version reported
# success having transferred nothing.
if [[ "$GO" -eq 1 ]]; then
  "${SSH[@]}" "$DROPLET" \
    "mkdir -p $REMOTE/app $REMOTE/scripts $REMOTE/deploy $REMOTE/baseline/models $REMOTE/data/parquet $REMOTE/data/search"
fi
echo "--- code ---"
"${RSYNC[@]}" "${CODE_EXCLUDES[@]}" app/ "$DROPLET:$REMOTE/app/"
"${RSYNC[@]}" "${CODE_EXCLUDES[@]}" scripts/features_seq.py scripts/taxonomy.py "$DROPLET:$REMOTE/scripts/"
"${RSYNC[@]}" requirements-server.txt "$DROPLET:$REMOTE/"
"${RSYNC[@]}" deploy/faffabout-web.service deploy/gunicorn.conf.py deploy/faffabout.nginx.conf \
  deploy/provision.sh "$DROPLET:$REMOTE/deploy/"
"${RSYNC[@]}" "${CODE_EXCLUDES[@]}" baseline/models/ "$DROPLET:$REMOTE/baseline/models/"

if [[ "$CODE_ONLY" -eq 0 ]]; then
  echo "--- data ---"
  # The local data/faffabout.duckdb is VIEWS over Parquet at absolute build-machine paths,
  # which resolve to nothing on the droplet. Ship the self-contained build under the name
  # the app opens. Rebuild it after any change to the archive tables.
  "${RSYNC[@]}" data/faffabout_serving.duckdb "$DROPLET:$REMOTE/data/faffabout.duckdb"
  DATA_EX=()
  [[ "$NO_INDEX" -eq 1 ]] && DATA_EX+=(--exclude 'archiveDB.idx*')
  "${RSYNC[@]}" ${DATA_EX[@]+"${DATA_EX[@]}"} data/search/ "$DROPLET:$REMOTE/data/search/"
fi

if [[ "$GO" -eq 1 && "$NO_RESTART" -eq 1 ]]; then
  echo
  echo "Transferred without restarting. Next: ssh to the droplet and run bash $REMOTE/deploy/provision.sh"
elif [[ "$GO" -eq 1 ]]; then
  echo
  echo "--- restart and verify ---"
  # rsync runs as root, so hand the tree back to the service user before restarting
  "${SSH[@]}" "$DROPLET" "chown -R faffabout:faffabout $REMOTE && systemctl restart $SERVICE && sleep 6 && systemctl is-active $SERVICE"
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
