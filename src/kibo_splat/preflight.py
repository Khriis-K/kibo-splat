"""Preflight: `kibo-splat-preflight` checks the build toolchain in seconds, no GPU needed.

Run it on the login node (or a debug node) after `source carc/env.sh`, before queueing a GPU job.
Fidelis lesson: a failed `module load` doesn't stop a job, so this catches it up front.
"""

import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from kibo_splat.smoke import Check, run_and_report

KERNEL = "__global__ void k() {}\nint main() { k<<<1, 1>>>(); }\n"


def check_nvcc() -> str:
    """nvcc is on PATH and shares torch's CUDA major version (torch refuses to build otherwise)."""
    if shutil.which("nvcc") is None:
        raise AssertionError("nvcc not on PATH; did `module load ver/2607 gcc/15.3.0 cuda/13.0.3` work?")
    import torch

    out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"release (\d+\.\d+)", out)
    if match is None:
        raise AssertionError(f"can't parse nvcc --version: {out!r}")
    release = match.group(1)
    if release.split(".")[0] != torch.version.cuda.split(".")[0]:
        raise AssertionError(f"nvcc is CUDA {release} but torch was built for CUDA {torch.version.cuda}")
    return f"nvcc {release}, torch built for CUDA {torch.version.cuda}"


def check_nvcc_compiles() -> str:
    """nvcc accepts the loaded host compiler: compiles a trivial kernel."""
    env = os.environ.copy()
    if sys.platform == "win32" and shutil.which("cl") is None:
        from kibo_splat.gsplat_loader import msvc_env

        env.update(msvc_env())
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "kernel.cu"
        source.write_text(KERNEL)
        result = subprocess.run(
            ["nvcc", "-std=c++17", "-c", str(source), "-o", str(Path(tmp) / "kernel.o")],
            capture_output=True,
            text=True,
            env=env,
        )
    if result.returncode != 0:
        raise AssertionError(f"nvcc failed on a trivial kernel:\n{result.stderr.strip()}")
    return "compiled a trivial kernel"


def check_python_headers() -> str:
    """The venv's Python ships Python.h, which torch's extension build includes."""
    include_dir = Path(sysconfig.get_paths()["include"])
    if not (include_dir / "Python.h").is_file():
        raise AssertionError(f"no Python.h in {include_dir}; use a uv-managed Python")
    return f"Python.h in {include_dir}"


def check_arch_list() -> str:
    """TORCH_CUDA_ARCH_LIST is set, so CPU-node builds match the GPU job's flags."""
    arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if not arch:
        raise AssertionError("TORCH_CUDA_ARCH_LIST unset; `source carc/env.sh` first")
    return f"TORCH_CUDA_ARCH_LIST={arch}, TORCH_EXTENSIONS_DIR={os.environ.get('TORCH_EXTENSIONS_DIR')}"


CHECKS: dict[str, Check] = {
    "nvcc": check_nvcc,
    "nvcc compile": check_nvcc_compiles,
    "python headers": check_python_headers,
    "arch list": check_arch_list,
}


def main() -> int:
    return run_and_report(CHECKS)
