# CARC notes

Distilled from Fidelis's `docs/carc.md` (C:\Projects\Fidelis\docs\carc.md), which has the job IDs and evidence.
Only what kibo-splat needs; no Snakemake.

## What we reuse, and why

| Choice | Why (Fidelis evidence) |
|---|---|
| torch `2.15.0+cu134`, uv-managed Python 3.12 | Passed on L40S/A100/A40 and the RTX 5070. System Python lacks `Python.h` → gsplat build failed. |
| gsplat fork `Khriis-K/gsplat@0270f88`, installed with `BUILD_NO_CUDA=1` | PyPI 1.5.3 doesn't build with CUDA 13; fork carries the fixes. JIT-compiles on first use. |
| `module load ver/2607 gcc/15.3.0 cuda/13.0.3` (that order) | nvcc must match torch's CUDA major (13). `cuda/13.0.3` refuses to load with `gcc/13.3.0`. |
| `import_gsplat()` pins build flags, `BUILD_3DGUT=0` | gsplat wipes its cache when flags change; 3DGUT compiles for hours. |
| One `TORCH_EXTENSIONS_DIR` per GPU type, explicit `TORCH_CUDA_ARCH_LIST` | Shared cache races / rebuilds when GPU types alternate. Explicit arch lets a CPU node prebuild. |
| **A40 as default GPU** (`--gres=gpu:a40:1`) | Only type that's had free cards; on both `gpu` and `debug`. L40S queues were very long. 48 GB is plenty. |
| Account via `SBATCH_ACCOUNT` etc. from `KIBOSPLAT_ACCOUNT` | The login's *default* account belongs to another group; a job without one charges them. |
| Preflight before every GPU job | A failed `module load` doesn't stop a job; it fails later, after the queue wait. |
| `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` on login nodes | 64-process cap; OpenBLAS thread spawn hung imports. |
| `uv sync` and gsplat build on a debug CPU node | Login nodes: 4 cores / 32 GB cap; exceeding it can lock you out. |

## Facts

- Login `discovery1.usc.edu` (VPN off campus). Data transfers: `hpc-transfer1.usc.edu`, in tmux. Compute nodes have no internet.
- Debug partition: 1 h max, A40 and P100 available. `gpu` partition: 48 h.
- SUs/min: A40/L40S/A100 = 8, 1 CPU = 1, 4 GB = 1. Charged for what you reserve.
- V100/P100 can't run this stack (CUDA 13 dropped sm_60–70) — so not the `tac450-th` reservation.
- Storage: `/scratch1/$USER/kibo-splat` for data, caches, run outputs (purged when the FS is >80% full — treat as disposable). Copy anything you want to keep (final `.ply`, charts) to `/project2/...` or your laptop.
- Free GPUs right now: `noderes -f -g`. Queue: `squeue --me`. History: `sacct -u $USER -S now-2days -X --format=JobID,JobName%15,State,Elapsed,NodeList`.
- "no kernel image is available" → the extension was built for another GPU; check `GPU_TYPE` / delete that `TORCH_EXTENSIONS_DIR`.
- Avoid VS Code Remote SSH (trips the process limit); use OnDemand or plain SSH.

## Open

- Which account this project charges (`KIBOSPLAT_ACCOUNT`).
- `gsplat` examples' reference trainer (the Day 2/3 safety net) has heavy extra deps (fused-ssim, pycolmap, nerfview, ...). Not installed yet; add only if Day 2 needs the fallback.
