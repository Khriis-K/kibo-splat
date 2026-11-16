"""Tests for the 2D Gaussian toy. Every expected value is worked out by hand, not by calling the code."""

import math

import pytest
import torch

from kibo_splat.toy2d import Gaussians2D, fit, mean_colour_image, psnr, render


def logit(p: float) -> float:
    """Inverse sigmoid, so a test can say 'opacity 0.8' in plain numbers."""
    return math.log(p / (1 - p))


def make_gaussians(means, scales, angles, colours, opacities) -> Gaussians2D:
    """Build Gaussians from human-readable values (pixels, radians, 0-1 colour and opacity)."""
    return Gaussians2D(
        means=torch.tensor(means, dtype=torch.float32),
        log_scales=torch.tensor(scales, dtype=torch.float32).log(),
        angles=torch.tensor(angles, dtype=torch.float32),
        colour_logits=torch.tensor([[logit(c) for c in rgb] for rgb in colours]),
        opacity_logits=torch.tensor([logit(o) for o in opacities]),
    )


def test_centre_pixel_is_opacity_times_colour():
    g = make_gaussians([[10.0, 7.0]], [[2.0, 2.0]], [0.0], [[0.9, 0.1, 0.1]], [0.8])
    image = render(g, height=16, width=20)
    assert image.shape == (16, 20, 3)
    # At the centre the Gaussian's falloff is exactly 1, so alpha = opacity; background is black.
    assert torch.allclose(image[7, 10], torch.tensor([0.72, 0.08, 0.08]), atol=1e-5)


def test_falloff_one_standard_deviation_away():
    g = make_gaussians([[10.0, 10.0]], [[2.0, 2.0]], [0.0], [[0.5, 0.5, 0.5]], [0.8])
    image = render(g, height=21, width=21)
    # 2 px right of centre with sigma 2 px is one standard deviation: falloff exp(-1/2).
    expected = 0.8 * math.exp(-0.5) * 0.5
    assert image[10, 12, 0].item() == pytest.approx(expected, abs=1e-5)


def test_rotation_turns_the_long_axis():
    # Long axis (sigma 4) starts along x; rotating 90 degrees should point it along y.
    g = make_gaussians([[10.0, 10.0]], [[4.0, 1.0]], [math.pi / 2], [[1.0 - 1e-6] * 3], [0.8])
    image = render(g, height=21, width=21)
    along_y = image[14, 10, 0].item()  # 4 px down = one sigma of the long axis
    along_x = image[10, 14, 0].item()  # 4 px right = four sigmas of the short axis
    assert along_y == pytest.approx(0.8 * math.exp(-0.5), abs=1e-4)
    assert along_x == pytest.approx(0.8 * math.exp(-8.0), abs=1e-4)


def test_front_to_back_compositing():
    # Two Gaussians on the same spot, both opacity 0.5. The first (red) is in front.
    g = make_gaussians(
        [[5.0, 5.0], [5.0, 5.0]],
        [[2.0, 2.0], [2.0, 2.0]],
        [0.0, 0.0],
        [[0.9, 0.1, 0.1], [0.1, 0.1, 0.9]],
        [0.5, 0.5],
    )
    pixel = render(g, height=11, width=11)[5, 5]
    # Red contributes 0.5 * colour; blue only gets the light red lets through: 0.5 * 0.5 * colour.
    red, blue = torch.tensor([0.9, 0.1, 0.1]), torch.tensor([0.1, 0.1, 0.9])
    assert torch.allclose(pixel, 0.5 * red + 0.25 * blue, atol=1e-5)


def test_psnr_known_value():
    # Every pixel off by 0.1 -> MSE 0.01 -> PSNR = 10 * log10(1 / 0.01) = 20 dB.
    a = torch.full((4, 4, 3), 0.5)
    b = torch.full((4, 4, 3), 0.6)
    assert psnr(a, b) == pytest.approx(20.0, abs=1e-3)


def test_psnr_identical_images_is_infinite():
    a = torch.rand(4, 4, 3, generator=torch.Generator().manual_seed(0))
    assert psnr(a, a) == math.inf


def checkerboard(size: int = 32, square: int = 8) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    on = ((ys // square + xs // square) % 2).float()
    return on[..., None].expand(size, size, 3).contiguous()


def test_mean_colour_image_scores_badly_on_a_checkerboard():
    # Null case: half black, half white -> mean 0.5 everywhere -> MSE 0.25 -> PSNR ~6.02 dB.
    target = checkerboard()
    baseline = mean_colour_image(target)
    assert torch.allclose(baseline, torch.full_like(target, 0.5))
    assert psnr(baseline, target) == pytest.approx(10 * math.log10(4), abs=1e-3)


def test_fit_beats_the_mean_colour_image():
    target = checkerboard()
    gaussians, losses = fit(target, num_gaussians=100, steps=300, seed=0)
    fitted = render(gaussians, height=32, width=32)
    assert losses[-1] < 0.5 * losses[0]
    # Null case: a working fit must clear the mean-colour image by a wide margin, not just tie it.
    baseline_psnr = psnr(mean_colour_image(target), target)
    assert psnr(fitted, target) > baseline_psnr + 10.0
