"""Tests for COLMAP's placement summary. Expected values are worked out by hand."""

import pytest

from kibo_splat.colmap_run import count_frames, summarize_placement


def test_count_frames_counts_only_images_in_the_folder_itself(tmp_path):
    # The placement denominator. Notes, calibration files and subfolders (the download ships an
    # empty .ipynb_checkpoints/) must not inflate it; upper-case extensions still count.
    for name in ["a.png", "b.PNG", "c.jpg", "d.jpeg", "gray.txt", "calib.yaml"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / ".ipynb_checkpoints").mkdir()
    (tmp_path / ".ipynb_checkpoints" / "e.png").write_bytes(b"")
    assert count_frames(tmp_path) == 4


def test_one_model_with_every_frame_is_full_placement():
    p = summarize_placement(model_sizes=[1068], total_frames=1068)
    assert p.placed == 1068
    assert p.rate == 1.0
    assert p.at_least_half


def test_split_reconstruction_counts_only_the_largest_model():
    # 400 + 300 = 700 frames is over half of 1,068, but the two models have unrelated
    # scales and origins, so only the 400 can train one splat together: 400 / 1068 < 0.5.
    p = summarize_placement(model_sizes=[300, 400], total_frames=1068)
    assert p.placed == 400
    assert p.n_models == 2
    assert p.rate == pytest.approx(400 / 1068)
    assert not p.at_least_half


def test_nothing_reconstructed_is_zero_placement():
    # Null case: COLMAP gave up entirely.
    p = summarize_placement(model_sizes=[], total_frames=1068)
    assert p.placed == 0
    assert p.n_models == 0
    assert p.rate == 0.0
    assert not p.at_least_half


def test_exactly_half_passes_the_at_least_half_rule():
    assert summarize_placement(model_sizes=[534], total_frames=1068).at_least_half
    assert not summarize_placement(model_sizes=[533], total_frames=1068).at_least_half


def test_more_placed_than_frames_is_an_error():
    # Means we counted the wrong image folder; a rate over 100% must not slip through.
    with pytest.raises(ValueError):
        summarize_placement(model_sizes=[1069], total_frames=1068)
