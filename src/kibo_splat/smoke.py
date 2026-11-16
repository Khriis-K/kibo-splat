"""GPU smoke test: `kibo-splat-smoke` prints one PASS/FAIL line per check.

Each check asserts against an independent ground truth, not just "it ran".
"""

from collections.abc import Callable

Check = Callable[[], str]


def check_torch() -> str:
    """A GPU matmul agrees with the same matmul on the CPU."""
    import torch

    a = torch.randn(512, 512, generator=torch.Generator().manual_seed(0))
    gpu = (a.cuda() @ a.cuda()).cpu()
    if not torch.allclose(gpu, a @ a, rtol=1e-2, atol=1e-1):  # loose: TF32 on Ampere+
        raise AssertionError("GPU matmul disagrees with CPU matmul")
    major, minor = torch.cuda.get_device_capability()
    return f"torch {torch.__version__} on {torch.cuda.get_device_name()} (sm_{major}{minor})"


def _render_one(means, size=64):
    """Render one white Gaussian with an identity camera; returns (H, W) brightness."""
    import torch

    from kibo_splat.gsplat_loader import import_gsplat

    gsplat = import_gsplat()
    colors, _, _ = gsplat.rasterization(
        means=means,
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        scales=torch.full((1, 3), 0.2, device="cuda"),
        opacities=torch.ones(1, device="cuda"),
        colors=torch.ones(1, 3, device="cuda"),
        viewmats=torch.eye(4, device="cuda")[None],
        Ks=torch.tensor([[[50.0, 0.0, size / 2], [0.0, 50.0, size / 2], [0.0, 0.0, 1.0]]], device="cuda"),
        width=size,
        height=size,
    )
    return colors[0, ..., 0], gsplat.__version__


def check_gsplat_forward() -> str:
    """One Gaussian straight ahead of the camera lands on the principal point."""
    import torch

    size = 64
    brightness, version = _render_one(torch.tensor([[0.0, 0.0, 5.0]], device="cuda"), size)
    row, col = divmod(int(brightness.argmax()), size)
    if abs(row - size / 2) > 1 or abs(col - size / 2) > 1:
        raise AssertionError(f"Gaussian landed at pixel ({row}, {col}), expected the center")
    if brightness[0, 0] >= brightness[row, col]:
        raise AssertionError("corner is as bright as the Gaussian")
    return f"gsplat {version}, peak at pixel ({row}, {col})"


def check_gsplat_backward() -> str:
    """Gradients flow: brightening the right half pulls the Gaussian's x position right (+x)."""
    import torch

    size = 64
    means = torch.tensor([[0.0, 0.0, 5.0]], device="cuda", requires_grad=True)
    brightness, _ = _render_one(means, size)
    brightness[:, size // 2 :].sum().backward()
    grad_x = float(means.grad[0, 0])
    if not grad_x > 0:
        raise AssertionError(f"d(right-half brightness)/dx = {grad_x}, expected > 0")
    return f"d(right-half)/dx = {grad_x:.3g} > 0"


CHECKS: dict[str, Check] = {
    "torch": check_torch,
    "gsplat forward": check_gsplat_forward,
    "gsplat backward": check_gsplat_backward,
}


def run_and_report(checks: dict[str, Check]) -> int:
    ok = True
    for name, check in checks.items():
        try:
            print(f"PASS  {name}: {check()}", flush=True)
        except Exception as exc:
            ok = False
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)
    return 0 if ok else 1


def main() -> int:
    return run_and_report(CHECKS)
