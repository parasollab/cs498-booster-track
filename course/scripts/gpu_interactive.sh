#!/usr/bin/env bash
# Start a short interactive GPU session and print the viser tunnel command.
# Usage: ./scripts/gpu_interactive.sh [hours]   (default 1, max 2)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/cluster.env

HOURS="${1:-1}"
if (( HOURS > 2 )); then
  echo "Interactive sessions are capped at 2h for this course." >&2
  echo "Longer work should be an sbatch job: scripts/train.sbatch" >&2
  exit 1
fi

echo "Requesting 1 GPU for ${HOURS}h on ${COURSE_PARTITION_INTERACTIVE} (account ${COURSE_ACCOUNT})..."
echo
echo "Once the session starts, note the hostname in your prompt (or run 'hostname')."
echo "To view viser from your laptop, run there:"
echo
echo "  ssh -J \$USER@gh-login.delta.ncsa.illinois.edu \\"
echo "      \$USER@<compute-node-hostname> -L 8080:localhost:8080"
echo
exec salloc \
  --account="${COURSE_ACCOUNT}" \
  --partition="${COURSE_PARTITION_INTERACTIVE}" \
  --nodes=1 --tasks=1 --cpus-per-task=16 \
  --gpus=1 \
  --time="${HOURS}:00:00" \
  srun --pty bash -c 'echo "=== GPU session on $(hostname) ==="; exec bash'
