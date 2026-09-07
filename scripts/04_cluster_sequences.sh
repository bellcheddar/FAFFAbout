#!/usr/bin/env bash
# 04_cluster_sequences.sh: cluster every TargetTrack sequence at 30% identity with MMseqs2.
#
# Every downstream split (Phase 2) is by cluster, never by row. The archive holds the
# same protein attempted across five or more orthologues by different centres, so a
# random row split leaks and hands you a meaningless 0.9 AUC.
#
# Input : data/parquet/targets.fasta   (written by 02_parse_tt_xml.py; one record per
#                                       distinct seq_md5, header = seq_md5)
# Output: data/clusters/tt30_cluster.tsv   (representative_md5 <TAB> member_md5)
#         data/clusters/tt30_rep_seq.fasta
#         data/clusters/tt30_all_seqs.fasta
#         data/clusters/tt30_clusters.parquet  (seq_md5, cluster_id, cluster_size)
#
# Usage: bash scripts/04_cluster_sequences.sh [--threads N]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTA="$ROOT/data/parquet/targets.fasta"
OUT="$ROOT/data/clusters"
TMP="$OUT/tmp"
PREFIX="$OUT/tt30"
THREADS="${THREADS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads) THREADS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v mmseqs >/dev/null || { echo "mmseqs not on PATH (brew install mmseqs2)" >&2; exit 1; }
[[ -s "$FASTA" ]] || { echo "missing $FASTA: run scripts/02_parse_tt_xml.py first" >&2; exit 1; }

mkdir -p "$OUT" "$TMP"
echo "mmseqs $(mmseqs version)  threads=$THREADS"
echo "input : $FASTA ($(grep -c '^>' "$FASTA") sequences)"

# --min-seq-id 0.30  : 30% identity threshold (spec section 4.4)
# -c 0.8 --cov-mode 1: 80% coverage of the shorter (target) sequence, so fragments and
#                      domain constructs of the same protein cluster with the full length
mmseqs easy-cluster "$FASTA" "$PREFIX" "$TMP" \
  --min-seq-id 0.30 -c 0.8 --cov-mode 1 \
  --threads "$THREADS" -v 1

rm -rf "$TMP"

# Resolve every seq_md5 to a cluster id and write Parquet for DuckDB.
"$ROOT/.venv/bin/python" - "$PREFIX" "$FASTA" <<'PY'
import sys
from pathlib import Path
import pandas as pd

prefix, fasta = sys.argv[1], sys.argv[2]
tsv = Path(f"{prefix}_cluster.tsv")
df = pd.read_csv(tsv, sep="\t", header=None, names=["cluster_rep", "seq_md5"], dtype=str)
n_in = sum(1 for line in open(fasta) if line.startswith(">"))
df["cluster_size"] = df.groupby("cluster_rep")["seq_md5"].transform("size")
# Stable integer cluster ids, largest clusters first so id 0 is the biggest family.
order = df.groupby("cluster_rep")["cluster_size"].first().sort_values(ascending=False)
cid = {rep: i for i, rep in enumerate(order.index)}
df["cluster_id"] = df["cluster_rep"].map(cid).astype("int32")
out = Path(f"{prefix}_clusters.parquet")
df[["seq_md5", "cluster_id", "cluster_rep", "cluster_size"]].to_parquet(out, index=False)
print(f"sequences in : {n_in:,}")
print(f"sequences out: {len(df):,}  (must match)")
print(f"clusters     : {len(cid):,}")
print(f"singletons   : {(order == 1).sum():,}")
print(f"largest      : {order.iloc[0]:,} members")
print(f"wrote {out}")
if len(df) != n_in:
    sys.exit("!! cluster file does not resolve every sequence")
PY
