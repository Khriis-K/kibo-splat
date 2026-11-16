# kibo-splat

A 3D Gaussian splat of **Kibō**, the Japanese module on the International Space Station,
built from NASA's Astrobee robot camera footage, plus an experiment on whether NASA's own
camera poses are good enough to train it.

A 3-day, hackathon-style learning project: one goal, understand 3D Gaussian splatting from the inside.

## Scope (locked)

| | |
|---|---|
| **Learn on** | Mip-NeRF 360 `garden`: the standard scene, with published numbers to check against |
| **NASA demo** | Astrobee ISS dataset, sequence `iva_kibo_trans`: 1,068 grayscale 1280×880 frames, camera travels 11.5 m through Kibō |
| **The twist** | Train the same images twice, once with **NASA's ground-truth poses** and once with **COLMAP's**, and compare quality and pose differences |
| **Out of scope** | Own CUDA rasterizer, colorization, dynamic scenes, other ISS modules, apps on top |

**Why the twist is a real question:** the one published Kibō splat
([arXiv 2608.21685](https://arxiv.org/html/2608.21685v1)) discarded Astrobee's onboard pose
estimates as too imprecise and recomputed everything with structure-from-motion. The public
dataset ships separate, refined ground-truth poses. Whether those are good enough for splatting is untested.

## Pipeline

See [docs/pipeline.mmd](docs/pipeline.mmd) (rendered: [docs/pipeline.svg](docs/pipeline.svg)).
Photos → camera poses (COLMAP *or* NASA ground truth) → initial Gaussians → training loop
(render with gsplat, loss, gradients, densify) → evaluation on held-out frames → web viewer.

## Data

| Dataset | Source | Notes |
|---|---|---|
| Mip-NeRF 360 `garden` | [Barron et al.](https://jonbarron.info/mipnerf360/) | COLMAP poses included |
| Astrobee `iva_kibo_trans` | [astrobee-iss-dataset.github.io](https://astrobee-iss-dataset.github.io), Kang et al., IEEE RA-L 2024 (NASA Ames + GIST) | Recorded 2022-05-13 in Kibō. Undistorted grayscale PNGs, intrinsics, ground-truth 6-DoF poses |

Data is not in the repo; `carc/download_data.sh` fetches both.
Why this sequence and not others: [docs/data-candidates/](docs/data-candidates/).

## Setup: CARC (once)

Full notes and why each step exists: [docs/carc.md](docs/carc.md).

1. In `~/.bashrc` on CARC: `export KIBOSPLAT_ACCOUNT=<your account>` (check `myaccount`).
2. Clone the repo onto CARC, then from a debug CPU node:
   ```bash
   salloc -p debug -c 8 --mem 32G -t 1:00:00
   bash carc/setup.sh          # uv sync, preflight, builds gsplat for A40 (~3-5 min)
   exit
   ```
3. GPU smoke test (login node, repo root):
   ```bash
   source carc/env.sh && sbatch carc/smoke.sbatch
   cat logs/splat-smoke_<jobid>.log   # expect 4 preflight PASS + 3 smoke PASS
   ```
4. Data, on `hpc-transfer1.usc.edu` in tmux: `bash carc/download_data.sh`

## Setup: local (RTX 5070, Windows)

torch `cu134` ships Blackwell (sm_120) kernels. Needs CUDA toolkit 13.x and VS 2022 C++ build tools.

```bash
uv sync
uv run kibo-splat-preflight
uv run kibo-splat-smoke        # first run compiles gsplat (~6 min)
```
