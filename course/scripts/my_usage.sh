#!/usr/bin/env bash
# How many GPU-hours have YOU used, and how is the course allocation doing?
#
#   ./scripts/my_usage.sh            # since COURSE_TERM_START (cluster.env)
#   ./scripts/my_usage.sh 2026-10-01 # since a specific date
#
# Works on any DeltaAI login node. Read-only: only queries Slurm accounting.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cluster.env

START="${1:-${COURSE_TERM_START:-$(date -d '-30 days' +%F)}}"

echo "=============================================================="
echo " Your GPU usage on account ${COURSE_ACCOUNT} since ${START}"
echo "=============================================================="

# Per-job history for this user (top-level jobs only, -X).
# AllocTRES contains e.g. "cpu=16,gres/gpu=1,mem=..."; multiply the gpu
# count by elapsed seconds to get GPU-hours.
sacct -X --user "$USER" --account "$COURSE_ACCOUNT" \
      --starttime "$START" --parsable2 --noheader \
      --format=JobID,JobName%20,State,ElapsedRaw,AllocTRES |
awk -F'|' '
  {
    gpus = 0
    if (match($5, /gres\/gpu[^,=]*=[0-9]+/)) {
      tres = substr($5, RSTART, RLENGTH)
      sub(/.*=/, "", tres)
      gpus = tres + 0
    }
    gpu_h = gpus * $4 / 3600.0
    total += gpu_h
    n += 1
    printf "  %-12s %-20s %-12s %6.2f GPU-h\n", $1, $2, $3, gpu_h
  }
  END {
    if (n == 0) print "  (no jobs found in this window)"
    printf "\n  YOUR TOTAL: %.2f GPU-hours across %d job(s)\n", total, n
  }'

echo
echo "=============================================================="
echo " Whole-course usage by user (shared allocation!)"
echo "=============================================================="
# Course-wide breakdown so you can see the shared budget burning.
sreport -t Hours -nP cluster AccountUtilizationByUser \
        account="$COURSE_ACCOUNT" start="$START" end=now 2>/dev/null |
awk -F'|' 'NF { printf "  %-16s %8s hours\n", ($3 == "" ? "(account total)" : $3), $5 }' ||
  echo "  (sreport unavailable — ask staff)"

echo
# NCSA's own balance tool, if present on this cluster.
if command -v accounts >/dev/null 2>&1; then
  echo "=== NCSA allocation balance (accounts) ==="
  accounts 2>/dev/null | grep -iE "${COURSE_ACCOUNT}|project|balance|remaining" || accounts
else
  echo "(NCSA 'accounts' balance tool not found on this node — using Slurm data above only)"
fi

echo
echo "Reminder: the allocation is shared by the whole class. If your total"
echo "is far above the HW's budget guidance, come to office hours BEFORE"
echo "submitting more jobs."
