# Day 1: can COLMAP place the Kibō frames? (issue #3)

**Verdict: model B goes ahead.** COLMAP placed more than half the frames in one model with default
sequential matching, so the "at least half" rule (`docs/SPEC.md`) is met. No matching retry is needed.

## What ran

- **Data:** `iva_kibo_trans`, 1,068 undistorted grayscale PNGs, 1280×880 (`data/astrobee/iva_kibo_trans/gray/`).
- **Camera:** one shared PINHOLE camera, 608.21 608.21 640 440, held fixed in bundle adjustment.
- **Matching:** sequential, COLMAP defaults (overlap 10, quadratic overlap on, no loop detection).
- **Mapper:** incremental, defaults apart from fixed intrinsics and `random_seed=0`.
- **Machine:** laptop, RTX 5070 Laptop GPU (sm_120). COLMAP 4.2.1 throughout.

**This run was done in two pieces, and the committed code doesn't reproduce it exactly:**

1. Feature extraction and the first 2,846 matched pairs came from an earlier version of
   `colmap_run.py` that used pip `pycolmap` (CPU only), with the same camera and matching settings.
   It was stopped once that was clearly going to take hours.
2. The current `colmap_run.py` then ran on the same database with the official CUDA `colmap.exe`.
   It skipped extraction (the database already existed), matched the remaining pairs on the GPU, and
   ran the mapper:
   `uv run kibo-splat-colmap data/astrobee/iva_kibo_trans/gray outputs/colmap_kibo_overlap10`.

A fresh run of the committed code would extract features with GPU SIFT, which gives slightly
different keypoints than CPU SIFT. So a rerun's numbers may differ a little. We don't expect it to
change the verdict, but that hasn't been tested.

## Result

| Measure | Value | Source |
|---|---|---|
| Frames placed in the largest model, when the verdict was called | **540 / 1,068 (50.6%)** | `run.log`, `num_reg_frames=540` at 17:41:12 |
| Models | 1 so far | `run.log`: no second model started |
| Verified image pairs | 9,657 | `database.db`, `two_view_geometries` |
| Mean SIFT keypoints per frame | 10,493 | `database.db`, `keypoints` |

**Caveat: this is a partial count, and the final placement rate is not known yet.** The mapper was
still running when the verdict was called. `num_reg_frames` is the live number of frames registered
in the model being built. COLMAP *can* de-register frames that bundle adjustment flags as bad, so
the count could in principle fall back under 534. That's unlikely: the count never dropped anywhere
in the log. But the margin was thin (540 against a threshold of 534). The log also shows 2 dense
Cholesky "linear solver failure" warnings at 14:44, steps the solver retried, not fatal errors.
The final number needs a complete run (see "Still open").

## How long it took (measured)

| Step | Where | Time | Source |
|---|---|---|---|
| Feature extraction + first 2,846 matched pairs | CPU (pip `pycolmap`) | ~37 min, then stopped | start and stop times of that run; pair count from `database.db` after stopping |
| The rest of matching | GPU (`colmap.exe`) | 274 s | `run.log` |
| Mapping to 540 frames | CPU | 3 h 47 min (13:54 → 17:41) | `run.log` |

Global bundle adjustment (a joint re-fit of every pose and point placed so far) dominates mapping,
and it mostly gets slower as the model grows (from `run.log`, frames registered when each pass started):

| Frames | 328 | 361 | 388 | 408 | 449 | 495 |
|---|---|---|---|---|---|---|
| Pass took | 12 min | 20 min | 17 min | 27 min | 29 min | 45 min |

GPU bundle adjustment isn't available in this build. In a separate 30-frame test with
`--Mapper.ba_use_gpu 1`, COLMAP warned that Ceres was compiled without CUDA or cuDSS support and
fell back to the CPU. That test's log isn't kept.

## Still open

- **The final placement rate** (the issue's "frames placed / 1,068"). It needs a mapping run that
  finishes. This run keeps its model in memory only (no snapshots), so a shutdown loses it. The next
  run should write snapshots (`--Mapper.snapshot_path`, `--Mapper.snapshot_frames_freq`) so it can resume.
- **Pose quality.** Placement says nothing about whether the poses are *right*. That is the Day 3
  pose comparison against NASA's ground truth (Sim(3) alignment, cm / degrees).
