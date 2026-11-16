"""2D Gaussian toy: fit a cloud of 2D Gaussians to one image.

This is the core loop of Gaussian splatting with the 3D parts (cameras, projection, depth) removed:
render the Gaussians into an image, compare it with the target, and let gradient descent nudge every
Gaussian's position, shape, colour and opacity. The 3D trainer will do the same thing after projecting
3D Gaussians onto the screen.

Run it with `uv run kibo-splat-toy2d`. Outputs go to outputs/toy2d/ (git-ignored).
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch

# Paths are anchored to the repo root (src/kibo_splat/toy2d.py -> up two levels), so the command
# works from any directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_SHEET = REPO_ROOT / "docs/data-candidates/kibo-iva_kibo_trans.jpg"
# The sheet is a 6x2 grid of Astrobee frames with a filename label above each; this box is the first frame.
FIRST_FRAME_BOX = (0, 14, 320, 234)  # (left, top, right, bottom) in sheet pixels
OUTPUT_DIR = REPO_ROOT / "outputs/toy2d"


@dataclass
class Gaussians2D:
    """N Gaussians, each described by 9 learned numbers.

    We never store size, colour or opacity directly. We store unconstrained values and map them into
    their valid range when rendering, so the optimizer can move them freely without ever producing
    a negative size or an opacity above 1.
    """

    means: torch.Tensor  # (N, 2) centre in pixels, as (x, y)
    log_scales: torch.Tensor  # (N, 2) log of the standard deviation along each axis; exp() keeps it > 0
    angles: torch.Tensor  # (N,) rotation of the first axis from the image's x-axis, in radians
    colour_logits: torch.Tensor  # (N, 3) RGB before a sigmoid squeezes it into 0-1
    opacity_logits: torch.Tensor  # (N,) opacity before a sigmoid squeezes it into 0-1


def render(g: Gaussians2D, height: int, width: int) -> torch.Tensor:
    """Draw the Gaussians into an (height, width, 3) image on a black background.

    Dense on purpose: every Gaussian is evaluated at every pixel, so all the maths sits in a few
    lines. Memory grows with Gaussians x pixels; gsplat's tiled CUDA rasterizer exists to avoid that.
    """
    device = g.means.device
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pixels = torch.stack([xs, ys], dim=-1)  # (H, W, 2), each pixel's (x, y)

    # Offset of every pixel from every Gaussian's centre: (N, H, W, 2).
    d = pixels[None] - g.means[:, None, None, :]

    # Express the offset in the Gaussian's own axes by rotating it back by the Gaussian's angle.
    cos = torch.cos(g.angles)[:, None, None]
    sin = torch.sin(g.angles)[:, None, None]
    u = cos * d[..., 0] + sin * d[..., 1]  # along the Gaussian's first axis
    v = -sin * d[..., 0] + cos * d[..., 1]  # along its second axis

    # Distance in standard deviations, then the bell-curve falloff: 1 at the centre, ~0 far away.
    scales = torch.exp(g.log_scales)
    falloff = torch.exp(-0.5 * ((u / scales[:, 0, None, None]) ** 2 + (v / scales[:, 1, None, None]) ** 2))

    # Sigmoid maps any real number into (0, 1): the valid range for opacity and colour.
    opacity = torch.sigmoid(g.opacity_logits)[:, None, None]
    colour = torch.sigmoid(g.colour_logits)  # (N, 3)
    alpha = opacity * falloff  # (N, H, W): how much of the light this Gaussian blocks at each pixel

    # Front-to-back alpha compositing, the same formula the 3D trainer uses. Gaussian i is seen
    # through everything in front of it: transmittance T_i = product over j < i of (1 - alpha_j).
    # A 2D image has no depth, so "in front" is simply a lower index.
    through = torch.cumprod(1 - alpha, dim=0)
    transmittance = torch.cat([torch.ones_like(through[:1]), through[:-1]], dim=0)
    weights = transmittance * alpha  # (N, H, W)
    return torch.einsum("nhw,nc->hwc", weights, colour)


def psnr(image: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB for images in 0-1. Higher is better; identical images give inf."""
    mse = torch.mean((image - target) ** 2).item()
    if mse == 0:
        return math.inf
    return 10 * math.log10(1 / mse)


def mean_colour_image(target: torch.Tensor) -> torch.Tensor:
    """The null-case baseline: every pixel set to the target's average colour."""
    return target.mean(dim=(0, 1), keepdim=True).expand_as(target).clone()


def init_gaussians(num: int, height: int, width: int, seed: int, device: str | torch.device) -> Gaussians2D:
    """Random starting point: centres spread over the image, grey, half-transparent, roughly tiling it."""
    gen = torch.Generator().manual_seed(seed)
    # Pick a size so that num Gaussians would roughly cover the image edge to edge.
    spacing = math.sqrt(height * width / num)

    def learnable(t: torch.Tensor) -> torch.Tensor:
        # requires_grad tells autograd to track this tensor, so loss.backward() fills in its gradient.
        return t.to(device).requires_grad_()

    return Gaussians2D(
        means=learnable(torch.rand(num, 2, generator=gen) * torch.tensor([width, height])),
        log_scales=learnable(torch.full((num, 2), math.log(spacing / 2))),
        angles=learnable(torch.rand(num, generator=gen) * math.pi),
        colour_logits=learnable(0.1 * torch.randn(num, 3, generator=gen)),  # near 0 -> sigmoid near 0.5 (grey)
        opacity_logits=learnable(torch.zeros(num)),  # 0 -> sigmoid 0.5 (half-transparent)
    )


def fit(target: torch.Tensor, num_gaussians: int, steps: int, seed: int = 0) -> tuple[Gaussians2D, list[float]]:
    """Fit Gaussians to an (H, W, 3) target in 0-1. Returns the Gaussians and the loss at each step."""
    height, width, _ = target.shape
    g = init_gaussians(num_gaussians, height, width, seed, target.device)
    # Adam moves each parameter by roughly lr per step, so positions (in pixels) get a pixel-sized
    # step while the logits and log-scales get a small one. The exact values (0.5, 0.05) are hand-picked
    # starting points, not tuned.
    optimizer = torch.optim.Adam(
        [
            {"params": [g.means], "lr": 0.5},
            {"params": [g.log_scales, g.angles, g.colour_logits, g.opacity_logits], "lr": 0.05},
        ]
    )
    losses = []
    for _ in range(steps):
        image = render(g, height, width)
        # L1: mean absolute error per pixel. 3DGS also adds an SSIM term; the toy keeps the simplest loss.
        loss = torch.mean(torch.abs(image - target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return g, losses


def load_target(width: int = 256) -> torch.Tensor:
    """First Astrobee frame from the Kibo contact sheet, resized to `width` px, as (H, W, 3) in 0-1."""
    # Imported here, not at the top: only the command-line run needs them, so the tests skip loading them.
    import numpy as np
    from PIL import Image

    frame = Image.open(TARGET_SHEET).convert("RGB").crop(FIRST_FRAME_BOX)
    height = round(frame.height * width / frame.width)
    frame = frame.resize((width, height), Image.LANCZOS)
    return torch.from_numpy(np.asarray(frame, dtype=np.float32) / 255.0)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = load_target().to(device)
    height, width, _ = target.shape
    num_gaussians, steps = 1000, 2000
    print(f"Fitting {num_gaussians} Gaussians to a {width}x{height} image for {steps} steps on {device}")

    g, losses = fit(target, num_gaussians, steps)
    with torch.no_grad():
        fitted = render(g, height, width).clamp(0, 1)
    baseline = mean_colour_image(target)
    fit_psnr = psnr(fitted, target)
    print(f"PSNR fit:         {fit_psnr:.2f} dB")
    print(f"PSNR mean colour: {psnr(baseline, target):.2f} dB  (null case)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set(xlabel="step", ylabel="L1 loss", yscale="log", title="2D toy: loss")
    fig.savefig(OUTPUT_DIR / "loss.png", dpi=120, bbox_inches="tight")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, img, title in zip(axes, [target, fitted, baseline], ["target", f"fit ({fit_psnr:.1f} dB)", "mean colour"]):
        ax.imshow(img.cpu().numpy())
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(OUTPUT_DIR / "side_by_side.png", dpi=120, bbox_inches="tight")
    print(f"Saved {OUTPUT_DIR / 'loss.png'} and {OUTPUT_DIR / 'side_by_side.png'}")


if __name__ == "__main__":
    main()
