#!/bin/bash
# One-time setup, run INSIDE a CPU debug allocation (not the login node: uv unpacks
# multi-GB CUDA wheels and gsplat's build uses 8 cores):
#
#   salloc -p debug -c 8 --mem 32G -t 1:00:00
#   cd <repo> && bash carc/setup.sh
#
# Builds gsplat for the A40 without a GPU (TORCH_CUDA_ARCH_LIST names the chip), so the
# first GPU job loads it in seconds instead of compiling for ~5 min on billed GPU time.
set -eo pipefail   # no -u: Lmod's module function trips over unset variables
cd "$(dirname "$0")/.."
source carc/env.sh

uv sync
uv run --no-sync kibo-splat-preflight

echo "Building gsplat for $GPU_TYPE (sm ${TORCH_CUDA_ARCH_LIST}) into $TORCH_EXTENSIONS_DIR ..."
time uv run --no-sync python -c "
from kibo_splat.gsplat_loader import import_gsplat
import_gsplat()
from gsplat.cuda._backend import _C
print('GSPLAT BUILT:', _C)
"
echo "Done. Next: sbatch carc/smoke.sbatch"
