"""Run COLMAP alone on the Kibo frames and report what share of them it placed.

This is the Day 1 go/no-go for model B (COLMAP's poses): if COLMAP can't place at least half the
frames, there is no model B to train. No splat training happens here.

Pipeline (structure from motion, SfM):
  1. Feature extraction: find SIFT keypoints in every frame.
  2. Sequential matching: match each frame against its 10 nearest neighbours in time, plus frames
     16, 32, ..., 512 apart (COLMAP's default "quadratic overlap"). Video frames overlap most with
     their neighbours, so this is far cheaper than matching every pair.
  3. Incremental mapping: start from one good pair, then add frames one by one, solving each new
     camera's pose and triangulating new 3D points, with bundle adjustment to tidy up.

Steps 1-3 run the official COLMAP binary (CUDA build), not pycolmap: the pip pycolmap wheel has no
CUDA, and CPU matching was far too slow (timings in docs/colmap-check.md). pycolmap is only used to
read the result. Re-running an interrupted run on the same out_dir reuses its database: extraction
is skipped and the matcher skips pairs it has already matched.

Usage (one out_dir per run, so two runs' models never get counted together):
  uv run kibo-splat-colmap data/astrobee/iva_kibo_trans/gray outputs/colmap_kibo_overlap10
  uv run kibo-splat-colmap data/astrobee/iva_kibo_trans/gray outputs/colmap_kibo_overlap30 --overlap 30
"""

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pycolmap

# Kibo camera (Astrobee NavCam, already undistorted): fx = fy = 608.21, cx = 640, cy = 440, 1280x880.
KIBO_PINHOLE = "608.21,608.21,640,440"

# The CUDA build from github.com/colmap/colmap releases (4.2.1, same version as our pycolmap),
# unpacked into the gitignored data/ folder. Override with COLMAP_EXE.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COLMAP_EXE = REPO_ROOT / "data/tools/colmap-cuda/bin/colmap.exe"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass
class Placement:
    placed: int  # frames in the largest model
    total: int  # frames we gave COLMAP
    n_models: int  # how many disconnected models COLMAP built

    @property
    def rate(self) -> float:
        return self.placed / self.total

    @property
    def at_least_half(self) -> bool:
        # The spec's rule: model B goes ahead only if COLMAP placed at least half the frames.
        return 2 * self.placed >= self.total


def summarize_placement(model_sizes: list[int], total_frames: int) -> Placement:
    """Turn the size of each model COLMAP built into one placement result.

    Only the largest model counts. When COLMAP loses track it starts a fresh model with its own
    scale and origin, and frames in different models can't train one splat together, so adding
    the models up would overstate what COLMAP recovered.
    """
    placed = max(model_sizes, default=0)
    if placed > total_frames:
        raise ValueError(f"{placed} frames placed but only {total_frames} given: wrong image folder?")
    return Placement(placed=placed, total=total_frames, n_models=len(model_sizes))


def count_frames(image_dir: Path) -> int:
    """Count the images COLMAP will see: the placement rate's denominator. Not subfolders."""
    return sum(1 for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES)


def colmap(step: str, **flags: object) -> None:
    """Run one COLMAP command and time it.

    COLMAP's flags contain dots (--Mapper.ba_refine_focal_length), which Python keyword names can't,
    so callers write a double underscore instead: Mapper__ba_refine_focal_length=0.
    """
    # Absolute path: Windows' CreateProcess can't find a relative path written with forward slashes.
    exe = Path(os.environ.get("COLMAP_EXE", DEFAULT_COLMAP_EXE)).resolve()
    cmd = [exe, step] + [f"--{name.replace('__', '.')}={value}" for name, value in flags.items()]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"{step}: {time.time() - t0:.0f} s", flush=True)


def run_colmap(image_dir: Path, out_dir: Path, overlap: int) -> Placement:
    out_dir.mkdir(parents=True, exist_ok=True)
    database = out_dir / "database.db"
    sparse = out_dir / "sparse"
    total = count_frames(image_dir)

    # One shared camera with the known intrinsics. They're fixed in the mapper too, so COLMAP only
    # has to find poses: that gives it its best chance, and keeps poses the only A/B difference.
    if not database.exists():
        colmap(
            "feature_extractor",
            database_path=database,
            image_path=image_dir,
            ImageReader__camera_model="PINHOLE",
            ImageReader__single_camera=1,
            ImageReader__camera_params=KIBO_PINHOLE,
        )
    colmap("sequential_matcher", database_path=database, SequentialMatching__overlap=overlap)

    sparse.mkdir(exist_ok=True)
    if any(sparse.iterdir()):
        # Old models would be counted alongside this run's. Use a new out_dir per run.
        raise SystemExit(f"{sparse} already holds models from an earlier run; pick a new out_dir")
    colmap(
        "mapper",
        database_path=database,
        image_path=image_dir,
        output_path=sparse,
        Mapper__ba_refine_focal_length=0,
        Mapper__ba_refine_principal_point=0,
        Mapper__ba_refine_extra_params=0,
        Mapper__random_seed=0,  # so a rerun gives the same answer
    )

    # The mapper writes each model it built to its own numbered folder: sparse/0, sparse/1, ...
    models = [pycolmap.Reconstruction(d) for d in sorted(sparse.iterdir()) if d.is_dir()]
    return summarize_placement([m.num_reg_images() for m in models], total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--overlap", type=int, default=10, help="neighbours each frame is matched with (default 10)")
    args = parser.parse_args()

    p = run_colmap(args.image_dir, args.out_dir, args.overlap)
    verdict = "model B goes ahead" if p.at_least_half else "under half: model B is off (one retry allowed)"
    print(f"placed {p.placed} / {p.total} frames = {p.rate:.1%} (largest of {p.n_models} models): {verdict}")


if __name__ == "__main__":
    main()
