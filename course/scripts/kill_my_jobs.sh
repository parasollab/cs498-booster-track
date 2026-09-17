#!/usr/bin/env bash
# Cancel your Slurm jobs — the emergency brake.
#
#   ./scripts/kill_my_jobs.sh              # cancel ALL your queued+running jobs (asks first)
#   ./scripts/kill_my_jobs.sh 12345 12346  # cancel only these job ids
#   ./scripts/kill_my_jobs.sh --pending    # cancel only jobs still waiting in the queue
#
# Use it when: you spot a bug right after submitting, a job is clearly stuck
# or diverged (reward NaN in the log), or you accidentally queued duplicates.
# Cancelling promptly is FREE GPU-hours back for the whole class — never let
# a known-bad job run to its wall-time limit out of politeness.
#
# Safety: only ever touches YOUR jobs (scancel is scoped to --user $USER).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cluster.env

MODE="all"
IDS=()
if [[ "${1:-}" == "--pending" ]]; then
  MODE="pending"
elif [[ $# -gt 0 ]]; then
  MODE="ids"
  IDS=("$@")
fi

show_targets() {
  case "$MODE" in
    all)     squeue --me --noheader --format="%.10i %.20j %.10T %.10M %.16R" ;;
    pending) squeue --me --states=PENDING --noheader --format="%.10i %.20j %.10T %.10M %.16R" ;;
    ids)     squeue --me --noheader --format="%.10i %.20j %.10T %.10M %.16R" \
               | grep -E "^\s*($(IFS='|'; echo "${IDS[*]}"))\s" || true ;;
  esac
}

TARGETS="$(show_targets)"
if [[ -z "$TARGETS" ]]; then
  echo "Nothing to cancel — you have no matching queued or running jobs."
  exit 0
fi

echo "About to CANCEL the following job(s):"
echo
echo "     JOBID                 NAME      STATE       TIME  NODE/REASON"
echo "$TARGETS"
echo
N=$(echo "$TARGETS" | wc -l)
read -r -p "Cancel ${N} job(s)? Type 'yes' to confirm: " REPLY
if [[ "$REPLY" != "yes" ]]; then
  echo "Aborted. Nothing was cancelled."
  exit 1
fi

case "$MODE" in
  all)     scancel --user "$USER" ;;
  pending) scancel --user "$USER" --state=PENDING ;;
  ids)     scancel "${IDS[@]}" ;;
esac

echo "Cancel request sent. Verifying..."
sleep 2
LEFT="$(show_targets)"
if [[ -z "$LEFT" ]]; then
  echo "Done — no matching jobs remain."
else
  echo "Still winding down (CG = completing, normal for a few seconds):"
  echo "$LEFT"
  echo "Re-run ./scripts/my_jobs.sh in a moment to confirm."
fi

echo
echo "Note: a cancelled training run keeps its already-saved checkpoints in"
echo "${COURSE_LOG_DIR}/rsl_rl/ — you lose only the un-run iterations."
