"""
Tests for compositing/compositor.py.

All synthetic, in-memory PreprocessedImage/ProcessedMask/LightingResult
objects — no model weights, external files, or network access needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_engine.compositing.compositor import Compositor
from ai_engine.exceptions import CompositingError
from ai_engine.schemas import (
    CompositingResult,
    LetterboxMetadata,
    LightingResult,
    PreprocessedImage,
    ProcessedMask,
)


@pytest.fixture
def compositor() -> Compositor:
    return Compositor()


def _make_preprocessed(width: int = 300, height: int = 200, color: int = 180) -> PreprocessedImage:
    return PreprocessedImage(
        original_image=np.full((height, width, 3), color, dtype=np.uint8),
        original_size=(width, height),
        model_input=np.zeros((512, 512, 3), dtype=np.uint8),
        letterbox=LetterboxMetadata(scale=1.0, pad_left=0, pad_top=0, pad_right=0, pad_bottom=0),
        source_format="JPEG",
    )


def _make_mask(width: int, height: int, region: tuple[int, int, int, int] | None = None) -> ProcessedMask:
    """`region` is (y0, y1, x0, x1) marked as floor; None means the
    whole frame is floor."""
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    if region is None:
        binary_mask[:, :] = 255
    else:
        y0, y1, x0, x1 = region
        binary_mask[y0:y1, x0:x1] = 255
    return ProcessedMask(
        binary_mask=binary_mask, area_ratio=1.0, bounding_box=(0, 0, width, height), discarded_region_count=0
    )


def _make_lighting_result(
    width: int,
    height: int,
    offset: tuple[int, int] = (0, 0),
    color: int = 40,
    alpha_value: int = 255,
) -> LightingResult:
    return LightingResult(
        adjusted_rgb=np.full((height, width, 3), color, dtype=np.uint8),
        alpha=np.full((height, width), alpha_value, dtype=np.uint8),
        canvas_size=(width, height),
        canvas_offset=offset,
        applied_gain=1.0,
        target_mean_luminance=128.0,
    )


# --- successful compositing --------------------------------------------


def test_successful_compositing_paints_material_where_expected(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(300, 200, color=180)
    mask = _make_mask(300, 200, region=(100, 190, 50, 250))
    lighting_result = _make_lighting_result(180, 80, offset=(60, 105), color=40)

    result = compositor.composite(preprocessed, mask, lighting_result)

    assert isinstance(result, CompositingResult)
    assert result.composited_image[150, 150, 0] < 100  # inside material+mask overlap
    assert result.composited_image[10, 10, 0] == 180  # untouched region stays room color


def test_applied_pixel_count_reflects_actual_overlap(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(300, 200)
    mask = _make_mask(300, 200, region=(100, 190, 50, 250))
    lighting_result = _make_lighting_result(180, 80, offset=(60, 105))

    result = compositor.composite(preprocessed, mask, lighting_result)

    assert result.applied_pixel_count == 180 * 80


# --- dimension preservation ------------------------------------------------


def test_output_dimensions_match_room_image(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(321, 213)
    mask = _make_mask(321, 213)
    lighting_result = _make_lighting_result(50, 50, offset=(10, 10))

    result = compositor.composite(preprocessed, mask, lighting_result)

    assert result.output_size == (321, 213)
    assert result.composited_image.shape == (213, 321, 3)


# --- mask handling (the critical "respect segmentation mask" guarantee) ----


def test_floor_mask_excludes_furniture_region_even_with_full_material_alpha(
    compositor: Compositor,
) -> None:
    preprocessed = _make_preprocessed(300, 200, color=180)
    binary_mask = np.zeros((200, 300), dtype=np.uint8)
    binary_mask[100:190, 50:250] = 255
    binary_mask[130:160, 120:180] = 0  # a "furniture" hole excluded from the floor
    mask = ProcessedMask(
        binary_mask=binary_mask, area_ratio=0.3, bounding_box=(50, 100, 200, 90), discarded_region_count=0
    )
    lighting_result = _make_lighting_result(180, 80, offset=(60, 105), color=40, alpha_value=255)

    result = compositor.composite(preprocessed, mask, lighting_result)

    # Over the furniture hole: material alpha says "paint here" but
    # the floor mask says "not floor" — the mask must win.
    assert result.composited_image[145, 150, 0] == 180
    # Elsewhere in the overlap, away from the hole: material paints normally.
    assert result.composited_image[170, 150, 0] < 100


def test_material_never_paints_outside_floor_mask_even_within_canvas_bounds(
    compositor: Compositor,
) -> None:
    preprocessed = _make_preprocessed(100, 100, color=200)
    mask = _make_mask(100, 100, region=None)  # start fully-floor
    # Shrink the mask to exclude the right half of the canvas region entirely.
    shrunk = mask.binary_mask.copy()
    shrunk[:, 50:] = 0
    mask = ProcessedMask(binary_mask=shrunk, area_ratio=0.5, bounding_box=(0, 0, 50, 100), discarded_region_count=0)
    lighting_result = _make_lighting_result(60, 60, offset=(20, 20), color=0, alpha_value=255)

    result = compositor.composite(preprocessed, mask, lighting_result)

    assert result.composited_image[40, 70, 0] == 200  # right half: excluded by mask, stays room color
    assert result.composited_image[40, 30, 0] == 0  # left half: painted


# --- alpha preservation / partial transparency ------------------------------


def test_partial_alpha_blends_proportionally(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(100, 100, color=180)
    mask = _make_mask(100, 100)
    lighting_result = _make_lighting_result(20, 20, offset=(40, 40), color=0, alpha_value=128)

    result = compositor.composite(preprocessed, mask, lighting_result)

    expected = 180 * (1 - 128 / 255)
    assert abs(int(result.composited_image[50, 50, 0]) - expected) < 2


def test_fully_transparent_material_leaves_room_untouched(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(100, 100, color=180)
    mask = _make_mask(100, 100)
    lighting_result = _make_lighting_result(20, 20, offset=(40, 40), color=0, alpha_value=0)

    result = compositor.composite(preprocessed, mask, lighting_result)

    assert result.composited_image[50, 50, 0] == 180
    assert result.applied_pixel_count == 0


# --- coordinate / offset edge cases -----------------------------------------


def test_negative_offset_is_clipped_not_crashed(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(100, 100, color=180)
    mask = _make_mask(100, 100)
    lighting_result = _make_lighting_result(50, 50, offset=(-10, -10), color=10, alpha_value=255)

    result = compositor.composite(preprocessed, mask, lighting_result)

    assert result.applied_pixel_count == 40 * 40  # only the visible 40x40 portion
    assert result.composited_image[5, 5, 0] == 10


def test_fully_out_of_bounds_canvas_leaves_room_unchanged(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(100, 100, color=180)
    mask = _make_mask(100, 100)
    lighting_result = _make_lighting_result(50, 50, offset=(1000, 1000), color=10, alpha_value=255)

    result = compositor.composite(preprocessed, mask, lighting_result)

    assert result.applied_pixel_count == 0
    assert np.array_equal(result.composited_image, preprocessed.original_image)


# --- invalid inputs ----------------------------------------------------------


def test_mismatched_mask_shape_raises_error(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(100, 100)
    mismatched_mask = _make_mask(50, 50)
    lighting_result = _make_lighting_result(20, 20, offset=(0, 0))

    with pytest.raises(CompositingError):
        compositor.composite(preprocessed, mismatched_mask, lighting_result)


def test_wrong_room_image_dtype_raises_error(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(100, 100)
    bad_preprocessed = PreprocessedImage(
        original_image=preprocessed.original_image.astype(np.float32),
        original_size=preprocessed.original_size,
        model_input=preprocessed.model_input,
        letterbox=preprocessed.letterbox,
        source_format=preprocessed.source_format,
    )
    mask = _make_mask(100, 100)
    lighting_result = _make_lighting_result(20, 20, offset=(0, 0))

    with pytest.raises(CompositingError):
        compositor.composite(bad_preprocessed, mask, lighting_result)


def test_canvas_size_mismatch_raises_error(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(100, 100)
    mask = _make_mask(100, 100)
    lighting_result = LightingResult(
        adjusted_rgb=np.full((20, 20, 3), 10, dtype=np.uint8),
        alpha=np.full((20, 20), 255, dtype=np.uint8),
        canvas_size=(999, 999),  # deliberately mismatched
        canvas_offset=(0, 0),
        applied_gain=1.0,
        target_mean_luminance=128.0,
    )

    with pytest.raises(CompositingError):
        compositor.composite(preprocessed, mask, lighting_result)


def test_mismatched_alpha_shape_raises_error(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(100, 100)
    mask = _make_mask(100, 100)
    lighting_result = LightingResult(
        adjusted_rgb=np.full((20, 20, 3), 10, dtype=np.uint8),
        alpha=np.full((5, 5), 255, dtype=np.uint8),  # mismatched
        canvas_size=(20, 20),
        canvas_offset=(0, 0),
        applied_gain=1.0,
        target_mean_luminance=128.0,
    )

    with pytest.raises(CompositingError):
        compositor.composite(preprocessed, mask, lighting_result)


def test_empty_room_image_dimension_raises_error(compositor: Compositor) -> None:
    bad_preprocessed = PreprocessedImage(
        original_image=np.zeros((0, 100, 3), dtype=np.uint8),
        original_size=(100, 0),
        model_input=np.zeros((512, 512, 3), dtype=np.uint8),
        letterbox=LetterboxMetadata(scale=1.0, pad_left=0, pad_top=0, pad_right=0, pad_bottom=0),
        source_format="JPEG",
    )
    mask = ProcessedMask(
        binary_mask=np.zeros((0, 100), dtype=np.uint8), area_ratio=0.0, bounding_box=(0, 0, 0, 0), discarded_region_count=0
    )
    lighting_result = _make_lighting_result(20, 20, offset=(0, 0))

    with pytest.raises(CompositingError):
        compositor.composite(bad_preprocessed, mask, lighting_result)


# --- deterministic output ---------------------------------------------------


def test_compositing_is_deterministic_across_repeated_calls(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(150, 120)
    mask = _make_mask(150, 120, region=(20, 100, 20, 130))
    lighting_result = _make_lighting_result(90, 70, offset=(25, 25))

    result_a = compositor.composite(preprocessed, mask, lighting_result)
    result_b = compositor.composite(preprocessed, mask, lighting_result)

    assert np.array_equal(result_a.composited_image, result_b.composited_image)
    assert result_a.applied_pixel_count == result_b.applied_pixel_count


# --- output schema / immutability -------------------------------------------


def test_result_schema_fields_are_present_and_typed(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(100, 100)
    mask = _make_mask(100, 100)
    lighting_result = _make_lighting_result(20, 20, offset=(0, 0))

    result = compositor.composite(preprocessed, mask, lighting_result)

    assert isinstance(result.composited_image, np.ndarray)
    assert isinstance(result.output_size, tuple)
    assert isinstance(result.applied_pixel_count, int)


def test_result_is_frozen(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(100, 100)
    mask = _make_mask(100, 100)
    lighting_result = _make_lighting_result(20, 20, offset=(0, 0))

    result = compositor.composite(preprocessed, mask, lighting_result)

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        result.applied_pixel_count = 0  # type: ignore[misc]


def test_room_image_input_is_not_mutated(compositor: Compositor) -> None:
    preprocessed = _make_preprocessed(100, 100, color=180)
    room_before = preprocessed.original_image.copy()
    mask = _make_mask(100, 100)
    lighting_result = _make_lighting_result(20, 20, offset=(0, 0), color=0)

    compositor.composite(preprocessed, mask, lighting_result)

    assert np.array_equal(preprocessed.original_image, room_before)
