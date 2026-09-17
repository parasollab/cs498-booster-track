#!/usr/bin/env bash
# Course sbatch template. THIS IS THE ONLY SANCTIONED WAY TO TRAIN.
#
#   sbatch scripts/train.sbatch <Task-Id> [extra args passed to `uv run train`]
#   e.g. sbatch scripts/train.sbatch Course-Cartpole-Swingup --env.scene.num-envs 4096
#
# Submit from the REPO ROOT. Everything this job produces stays inside the
# repo, under logs/:
#   logs/slurm-<jobid>.out                        this job's stdout/stderr
#   logs/rsl_rl/<experiment>/<run>/model_<n>.pt   checkpoints
#   logs/rsl_rl/<experiment>/<run>/params/        resolved env + agent cfg
# <experiment> comes from the task's runner cfg, and is what keeps assignments
# apart in one shared log root: "hw0_cartpole" for Course-Cartpole-*,
# "cartpole_double" for Course-Cartpole-Double-*, "hw2_reinforce" and
# "hw2_reinforce_cartpole" for Course-HW2-*.
#
# CHANGED IN HW2: this used to be hw0/logs/. One log root now serves every
# assignment. Runs you made before the change are still where you left them.
#
# The --output path below is relative to the directory you submit from and
# Slurm will NOT create it, which is why logs/ is committed to the repo.
#
# Resource limits are fixed on purpose. Do not add your own #SBATCH lines.
#
#SBATCH --job-name=course-train
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --account=bign-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

TASK_ID="${1:?Usage: sbatch scripts/train.sbatch <Task-Id> [train args...]}"
shift

if [[ ! -f scripts/train.sbatch ]]; then
  echo "ERROR: submit from the repo root (found no scripts/train.sbatch in ${SLURM_SUBMIT_DIR})" >&2
  exit 1
fi

LOG_DIR="${SLURM_SUBMIT_DIR}/logs"
LOG_ROOT="${LOG_DIR}/rsl_rl"
mkdir -p "${LOG_ROOT}"

# Keep W&B's scratch files next to the run instead of in $HOME.
export WANDB_DIR="${LOG_DIR}"

# Inject the course log root unless the caller passed their own --log-root
# (tyro errors out on a duplicate flag).
LOG_ROOT_ARGS=(--log-root "${LOG_ROOT}")
for arg in "$@"; do
  if [[ "${arg}" == "--log-root" || "${arg}" == --log-root=* ]]; then
    LOG_ROOT_ARGS=()
    break
  fi
done

echo "== $(date) job ${SLURM_JOB_ID} on $(hostname) =="
echo "== task: ${TASK_ID}  extra args: $*"
echo "== log root: ${LOG_ROOT_ARGS[*]:-(caller-supplied)}"
nvidia-smi || true

uv run train "${TASK_ID}" "${LOG_ROOT_ARGS[@]}" "$@"

echo "== $(date) done =="
