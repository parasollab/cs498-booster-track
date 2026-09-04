#!/usr/bin/env bash
# Are all your training jobs done? Shows active jobs, recent history, and a
# clear verdict, with log paths for anything that failed.
#
#   ./scripts/my_jobs.sh        # history since COURSE_TERM_START (cluster.env)
#   ./scripts/my_jobs.sh 2d     # history for the last 2 days (or 12h, 1w...)
#
# Read-only: only queries Slurm.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cluster.env

WINDOW="${1:-}"
if [[ -n "$WINDOW" ]]; then
  # Accept 12h / 2d / 1w style windows.
  case "$WINDOW" in
    *h) START=$(date -d "-${WINDOW%h} hours" +%FT%T) ;;
    *d) START=$(date -d "-${WINDOW%d} days" +%FT%T) ;;
    *w) START=$(date -d "-$(( ${WINDOW%w} * 7 )) days" +%FT%T) ;;
    *)  START="$WINDOW" ;;  # assume a date
  esac
else
  START="${COURSE_TERM_START:-$(date -d '-7 days' +%F)}"
fi

echo "================ ACTIVE (queued or running) ================"
ACTIVE=$(squeue --me --noheader \
  --format="%.10i %.20j %.10T %.10M %.10l %.16R" || true)
if [[ -n "$ACTIVE" ]]; then
  echo "     JOBID                 NAME      STATE       TIME  TIME_LIMIT  NODE/REASON"
  echo "$ACTIVE"
else
  echo "  (none)"
fi

echo
echo "================ HISTORY since ${START} ================"
sacct -X --user "$USER" --starttime "$START" --parsable2 --noheader \
      --format=JobID,JobName%20,State,Elapsed,ExitCode |
awk -F'|' '
  { printf "  %-12s %-20s %-14s %-10s exit %s\n", $1, $2, $3, $4, $5
    st = $3
    sub(/ .*/, "", st)                 # "CANCELLED by ..." -> CANCELLED
    count[st] += 1
    if (st != "COMPLETED" && st != "RUNNING" && st != "PENDING") bad[$1] = st
  }
  END {
    if (NR == 0) { print "  (no jobs in this window)"; exit }
    printf "\n  Summary:"
    for (s in count) printf " %s=%d", s, count[s]
    print ""
    for (j in bad)
      printf "  !! job %s ended %s — check its log:\n     less %s/slurm-%s.out\n", \
             j, bad[j], ENVIRON["COURSE_LOG_DIR"], j
  }'

echo
echo "================ VERDICT ================"
if [[ -z "$ACTIVE" ]]; then
  echo "  All submitted jobs are DONE (nothing queued or running)."
  echo "  Checkpoints, if any, are under: ${COURSE_LOG_DIR}/rsl_rl/"
else
  N=$(echo "$ACTIVE" | wc -l)
  echo "  ${N} job(s) still active. Check again later, or watch one live:"
  echo "    tail -f ${COURSE_LOG_DIR}/slurm-<jobid>.out"
fi
