# Source this in every CARC shell and job:  source carc/env.sh
# Sets the Slurm account, modules, caches and gsplat build target in one place.

# --- Account: set KIBOSPLAT_ACCOUNT in ~/.bashrc. Never rely on the default account
# (on this user's CARC login it belongs to another group). Slurm reads these env vars
# for sbatch / salloc / srun, so no command needs --account.
if [ -z "${KIBOSPLAT_ACCOUNT:-}" ]; then
    echo "env.sh: KIBOSPLAT_ACCOUNT is not set (add 'export KIBOSPLAT_ACCOUNT=<account>' to ~/.bashrc)" >&2
    return 1 2>/dev/null || exit 1
fi
export SBATCH_ACCOUNT="$KIBOSPLAT_ACCOUNT" SALLOC_ACCOUNT="$KIBOSPLAT_ACCOUNT" SLURM_ACCOUNT="$KIBOSPLAT_ACCOUNT"

# --- Paths. Rule from Fidelis: cheap to rebuild -> scratch; expensive -> project2.
export KIBOSPLAT_SCRATCH="/scratch1/$USER/kibo-splat"
export KIBOSPLAT_DATA="$KIBOSPLAT_SCRATCH/data"
export UV_CACHE_DIR="$KIBOSPLAT_SCRATCH/uv-cache"
export UV_LINK_MODE=copy            # cache and venv are on different file systems
mkdir -p "$KIBOSPLAT_DATA" "$UV_CACHE_DIR"
[ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"   # puts uv on PATH

# --- Toolchain. cuda/13.0.3 only loads after ver/2607 and gcc/15.3.0, in that order.
module purge
if ! module load ver/2607 gcc/15.3.0 cuda/13.0.3; then
    echo "env.sh: module load failed" >&2
    return 1 2>/dev/null || exit 1
fi

# --- gsplat build target. One build cache per GPU type: concurrent or alternating builds
# for different GPUs in one cache race or rebuild every time. Default: A40 (sm_86).
export GPU_TYPE="${GPU_TYPE:-a40}"
case "$GPU_TYPE" in
    a40)  export TORCH_CUDA_ARCH_LIST="8.6" ;;
    l40s) export TORCH_CUDA_ARCH_LIST="8.9" ;;
    a100) export TORCH_CUDA_ARCH_LIST="8.0" ;;
    *) echo "env.sh: unsupported GPU_TYPE=$GPU_TYPE (v100/p100 can't run CUDA 13)" >&2
       return 1 2>/dev/null || exit 1 ;;
esac
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions_kibo_splat_$GPU_TYPE"

# --- Login nodes cap you at 64 processes; BLAS/OpenMP thread pools blow through it.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
fi
