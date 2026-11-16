#!/bin/bash
# GPU smoke test on one A40 (debug partition: 1 h max, usually short queue).
# Submit from the repo root:  sbatch carc/smoke.sbatch
# Cost cap: (8 GPU + 4 CPU + 4 mem) SUs/min x 30 min = 480 SUs. Account comes from SBATCH_ACCOUNT.
#SBATCH --job-name=splat-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log
set -eo pipefail   # no -u: Lmod's module function trips over unset variables
cd "$SLURM_SUBMIT_DIR"
export GPU_TYPE=a40
source carc/env.sh

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv run --no-sync kibo-splat-preflight
uv run --no-sync kibo-splat-smoke
