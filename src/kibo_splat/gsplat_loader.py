"""Import gsplat with the environment its JIT build needs (carried over from Fidelis).

- gsplat wipes its build cache whenever its build flags change, so they're pinned here.
  3DGUT (fisheye / rolling-shutter cameras) is off: we don't need it and it compiles for hours.
- On Windows, torch looks up MSVC's linker on every load, so MSVC must be on PATH.

Always `from kibo_splat.gsplat_loader import import_gsplat` instead of `import gsplat`.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

VSWHERE = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
    r"Microsoft Visual Studio\Installer\vswhere.exe"
)

# Every env var gsplat's build reads; None means "leave unset" (gsplat's default).
BUILD_FLAGS = {
    "BUILD_3DGUT": "0",
    "DEBUG": "0",
    "FAST_MATH": "1",
    "WITH_SYMBOLS": "0",
    "NVCC_FLAGS": "",
    "BUILD_CAMERA_WRAPPERS": "0",
    "BUILD_3DGS": None,
    "BUILD_2DGS": None,
    "BUILD_ADAM": None,
    "BUILD_RELOC": None,
    "BUILD_LOSSES": None,
    "NUM_CHANNELS": None,
}


def import_gsplat() -> ModuleType:
    for name, value in BUILD_FLAGS.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    if sys.platform == "win32" and shutil.which("cl") is None:
        os.environ.update(msvc_env())

    import gsplat  # noqa: TID251 -- the one sanctioned import

    return gsplat


def msvc_env() -> dict[str, str]:
    """The environment a VS developer shell would have: the newest VS with C++ tools."""
    if not VSWHERE.exists():
        raise RuntimeError(f"Visual Studio Installer not found at {VSWHERE}; gsplat needs MSVC to load")
    install = subprocess.run(
        [
            str(VSWHERE),
            *["-latest", "-products", "*", "-property", "installationPath"],
            *["-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"],
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not install:
        raise RuntimeError("No Visual Studio with C++ build tools found; gsplat needs MSVC to load")
    vcvars = Path(install) / r"VC\Auxiliary\Build\vcvars64.bat"
    dump = subprocess.run(f'"{vcvars}" >nul && set', shell=True, capture_output=True, text=True, check=True).stdout
    return dict(line.split("=", 1) for line in dump.splitlines() if "=" in line)
