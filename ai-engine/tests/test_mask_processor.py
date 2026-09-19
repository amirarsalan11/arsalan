"""
Tests for mask_processing/mask_processor.py.

All synthetic — no real segmentation output is needed. Every test
builds its own minimal 512x512 float32 probability mask directly.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ai_engine.mask_processing.mask_processor import MaskProcessor
from ai_engine.schemas import LetterboxMetadata


@pytest.fixture
def mask_processor() -> MaskProcessor:
    return MaskProcessor()


@pytest.fixture
def identity_letterbox() -> LetterboxMetadata:
    """A letterbox with no padding at all — original image was
    already square, scale=1.0. Simplifies tests that don't care about
    the inverse-letterbox step itself."""
    return LetterboxMetadata(scale=1.0, pad_left=0, pad_top=0, pad_right=0, pad_bottom=0)


def test_output_is_strictly_binary(mask_processor: MaskProcessor, identity_letterbox: LetterboxMetadata) -> None:
    raw_mask = np.zeros((512, 512), dtype=np.float32)
    raw_mask[100:400, 100:400] = 0.9

    result = mask_processor.process(raw_mask, (512, 512), identity_letterbox)

    assert result.binary_mask.dtype == np.uint8
    assert set(np.unique(result.binary_mask).tolist()).issubset({0, 255})


def test_inverse_letterbox_crops_padding_before_upscaling(mask_processor: MaskProcessor) -> None:
    # 800x600 original -> scale 0.64, vertical padding of 64px top/bottom
    # (matches the Preprocessor's own letterbox math for this aspect ratio).
    letterbox = LetterboxMetadata(scale=0.64, pad_left=0, pad_top=64, pad_right=0, pad_bottom=64)

    raw_mask = np.zeros((512, 512), dtype=np.float32)
    # Confident floor filling the entire cropped (non-padded) region.
    raw_mask[64:448, :] = 0.9

    result = mask_processor.process(raw_mask, (800, 600), letterbox)

    assert result.binary_mask.shape == (600, 800)
    # The whole original-resolution image should end up foreground,
    # since the entire cropped region (i.e. everything that wasn't
    # letterbox padding) was confidently floor.
    assert result.area_ratio > 0.95


def test_area_ratio_reflects_fraction_of_frame(
    mask_processor: MaskProcessor, identity_letterbox: LetterboxMetadata
) -> None:
    raw_mask = np.zeros((512, 512), dtype=np.float32)
    raw_mask[128:384, 128:384] = 0.9  # a 256x256 square within a 512x512 frame -> 25%

    result = mask_processor.process(raw_mask, (512, 512), identity_letterbox)

    assert 0.20 < result.area_ratio < 0.30


def test_morphological_opening_removes_small_noise_specks(
    mask_processor: MaskProcessor, identity_letterbox: LetterboxMetadata
) -> None:
    raw_mask = np.zeros((512, 512), dtype=np.float32)
    raw_mask[100:400, 100:400] = 0.9  # main blob
    # A single-pixel-scale noise speck, far too small to survive
    # opening, and disconnected from the main blob.
    raw_mask[10:12, 10:12] = 0.9

    result = mask_processor.process(raw_mask, (512, 512), identity_letterbox)

    # The speck must not survive as its own component (either removed
    # by opening, or discarded as a smaller connected component).
    assert result.binary_mask[10, 10] == 0 or result.discarded_region_count >= 1


def test_hole_filling_fills_enclosed_gap(
    mask_processor: MaskProcessor, identity_letterbox: LetterboxMetadata
) -> None:
    raw_mask = np.zeros((512, 512), dtype=np.float32)
    raw_mask[100:400, 100:400] = 0.9
    raw_mask[240:260, 240:260] = 0.0  # enclosed hole, fully surrounded by floor

    result = mask_processor.process(raw_mask, (512, 512), identity_letterbox)

    assert result.binary_mask[250, 250] == 255


def test_hole_filling_does_not_fill_border_connected_background(
    mask_processor: MaskProcessor, identity_letterbox: LetterboxMetadata
) -> None:
    # An "L"-shaped/ring floor region around a central gap that is
    # itself connected to the image border via a corridor — this gap
    # must NOT be filled, since it's reachable from outside, unlike a
    # truly enclosed hole.
    raw_mask = np.zeros((512, 512), dtype=np.float32)
    raw_mask[50:450, 50:450] = 0.9  # big floor square
    raw_mask[150:350, 150:350] = 0.0  # big central gap
    raw_mask[150:350, 0:150] = 0.0  # corridor connecting the gap to the left border (outside the floor square too)

    result = mask_processor.process(raw_mask, (512, 512), identity_letterbox)

    # Center of the big gap should remain background, since it's
    # reachable from the image border through the corridor.
    assert result.binary_mask[250, 250] == 0


def test_multiple_disconnected_regions_keeps_only_largest(
    mask_processor: MaskProcessor, identity_letterbox: LetterboxMetadata
) -> None:
    raw_mask = np.zeros((512, 512), dtype=np.float32)
    raw_mask[100:400, 100:400] = 0.9  # large main blob (300x300 = 90000 px)
    raw_mask[10:40, 470:500] = 0.9  # separate small blob (30x30 = 900 px), far from main blob

    result = mask_processor.process(raw_mask, (512, 512), identity_letterbox)

    assert result.discarded_region_count >= 1
    # The small, separate blob's location should be background in the result.
    assert result.binary_mask[25, 485] == 0
    # The main blob's location should still be foreground.
    assert result.binary_mask[250, 250] == 255


def test_discarded_region_count_is_zero_for_single_region(
    mask_processor: MaskProcessor, identity_letterbox: LetterboxMetadata
) -> None:
    raw_mask = np.zeros((512, 512), dtype=np.float32)
    raw_mask[100:400, 100:400] = 0.9

    result = mask_processor.process(raw_mask, (512, 512), identity_letterbox)

    assert result.discarded_region_count == 0


def test_empty_mask_produces_zero_area_ratio(
    mask_processor: MaskProcessor, identity_letterbox: LetterboxMetadata
) -> None:
    raw_mask = np.zeros((512, 512), dtype=np.float32)

    result = mask_processor.process(raw_mask, (512, 512), identity_letterbox)

    assert result.area_ratio == 0.0
    assert result.bounding_box == (0, 0, 0, 0)
    assert result.discarded_region_count == 0


def test_bounding_box_matches_foreground_extent(
    mask_processor: MaskProcessor, identity_letterbox: LetterboxMetadata
) -> None:
    raw_mask = np.zeros((512, 512), dtype=np.float32)
    raw_mask[100:200, 150:250] = 0.9  # a 100x100 block

    result = mask_processor.process(raw_mask, (512, 512), identity_letterbox)

    x, y, w, h = result.bounding_box
    # Allow a small tolerance for morphological/bilateral smoothing
    # shrinking or growing the boundary by a few pixels.
    assert abs(x - 150) <= 10
    assert abs(y - 100) <= 10
    assert abs(w - 100) <= 20
    assert abs(h - 100) <= 20


def test_edge_smoothing_preserves_strictly_binary_output(
    mask_processor: MaskProcessor, identity_letterbox: LetterboxMetadata
) -> None:
    # A deliberately jagged mask (checkerboard-ish edge) to exercise
    # the bilateral smoothing + re-threshold path meaningfully.
    raw_mask = np.zeros((512, 512), dtype=np.float32)
    raw_mask[100:400, 100:400] = 0.9
    rng = np.random.default_rng(42)
    jagged_edge = rng.random((300, 20)) > 0.5
    raw_mask[100:400, 390:410] = np.where(jagged_edge, 0.9, 0.0)

    result = mask_processor.process(raw_mask, (512, 512), identity_letterbox)

    assert set(np.unique(result.binary_mask).tolist()).issubset({0, 255})
