"""
Tests for lighting/lighting_processor.py.

All synthetic, in-memory PerspectiveTransformResult objects — no model
weights, external files, or network access needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_engine.exceptions import LightingAdjustmentError
from ai_engine.lighting.lighting_processor import (
    _DEFAULT_TARGET_MEAN_LUMINANCE,
    _MAX_GAIN,
    _MIN_GAIN,
    LightingProcessor,
)
from ai_engine.schemas import LightingResult, PerspectiveTransformResult


@pytest.fixture
def lighting_processor() -> LightingProcessor:
    return LightingProcessor()


def _make_perspective_result(
    rgb: np.ndarray,
    alpha: np.ndarray,
    canvas_offset: tuple[int, int] = (5, 7),
) -> PerspectiveTransformResult:
    height, width = rgb.shape[:2]
    return PerspectiveTransformResult(
        warped_rgb=rgb,
        warped_alpha=alpha,
        homography_matrix=np.eye(3, dtype=np.float64),
        canvas_size=(width, height),
        canvas_offset=canvas_offset,
        destination_corners=((0, 0), (width, 0), (width, height), (0, height)),
        source_size=(width, height),
        source_has_alpha=True,
    )


def _solid_material(color: int, width: int = 100, height: int = 50, alpha_value: int = 255):
    rgb = np.full((height, width, 3), color, dtype=np.uint8)
    alpha = np.full((height, width), alpha_value, dtype=np.uint8)
    return rgb, alpha


# --- 1/2: valid RGB / RGBA (alpha-bearing) input --------------------------


def test_valid_rgb_opaque_input(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(100)
    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert isinstance(result, LightingResult)
    assert result.adjusted_rgb.dtype == np.uint8


def test_valid_rgba_with_real_transparency(lighting_processor: LightingProcessor) -> None:
    rgb = np.full((50, 100, 3), 100, dtype=np.uint8)
    alpha = np.zeros((50, 100), dtype=np.uint8)
    alpha[10:40, 20:80] = 255  # partial transparency: only part of the canvas is opaque

    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert result.adjusted_rgb.shape == rgb.shape
    assert result.alpha.shape == alpha.shape


# --- 3: output dimensions unchanged ---------------------------------------


def test_output_dimensions_match_input(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(100, width=137, height=61)
    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert result.adjusted_rgb.shape == rgb.shape
    assert result.alpha.shape == alpha.shape


# --- 4: alpha preservation -------------------------------------------------


def test_alpha_is_passed_through_unchanged(lighting_processor: LightingProcessor) -> None:
    rgb = np.full((50, 100, 3), 40, dtype=np.uint8)
    alpha = np.zeros((50, 100), dtype=np.uint8)
    alpha[10:40, 20:80] = 200  # a specific, non-binary value to confirm exact passthrough

    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert np.array_equal(result.alpha, alpha)


def test_fully_transparent_regions_stay_fully_transparent(
    lighting_processor: LightingProcessor,
) -> None:
    rgb, _ = _solid_material(10)
    alpha = np.zeros((50, 100), dtype=np.uint8)
    alpha[10:40, 20:80] = 255

    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert np.all(result.alpha[alpha == 0] == 0)


# --- 5: pixel values remain valid ------------------------------------------


def test_pixel_values_remain_in_valid_uint8_range(lighting_processor: LightingProcessor) -> None:
    # Near-white material with a low target would otherwise risk
    # clipping issues if gain math were wrong.
    rgb, alpha = _solid_material(250)
    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert result.adjusted_rgb.dtype == np.uint8
    assert result.adjusted_rgb.min() >= 0
    assert result.adjusted_rgb.max() <= 255


def test_extreme_dark_input_does_not_overflow_or_wrap(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(1)
    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert result.adjusted_rgb.max() <= 255
    assert result.adjusted_rgb.min() >= 0


# --- 6: deterministic output ------------------------------------------------


def test_deterministic_across_repeated_calls(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(77)
    perspective_result = _make_perspective_result(rgb, alpha)

    result_a = lighting_processor.process(perspective_result)
    result_b = lighting_processor.process(perspective_result)

    assert np.array_equal(result_a.adjusted_rgb, result_b.adjusted_rgb)
    assert result_a.applied_gain == result_b.applied_gain


# --- 7/8/9: dark / bright / neutral input behavior ---------------------------


def test_dark_input_is_brightened_toward_target(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(30)
    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert result.applied_gain > 1.0
    assert result.adjusted_rgb[25, 50, 0] > 30


def test_bright_input_is_darkened_toward_target(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(220)
    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert result.applied_gain < 1.0
    assert result.adjusted_rgb[25, 50, 0] < 220


def test_neutral_input_near_default_target_needs_little_adjustment(
    lighting_processor: LightingProcessor,
) -> None:
    rgb, alpha = _solid_material(int(_DEFAULT_TARGET_MEAN_LUMINANCE))
    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert abs(result.applied_gain - 1.0) < 0.05


def test_gain_is_clamped_for_pathologically_dark_input(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(2)
    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert result.applied_gain <= _MAX_GAIN


def test_gain_is_clamped_for_pathologically_bright_input(
    lighting_processor: LightingProcessor,
) -> None:
    rgb, alpha = _solid_material(254)
    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert result.applied_gain >= _MIN_GAIN


def test_border_transparent_pixels_excluded_from_luminance_statistics(
    lighting_processor: LightingProcessor,
) -> None:
    """Perspective's borderValue=0 fills RGB=0/alpha=0 outside the
    floor quad — that padding must not corrupt the exposure
    statistic. Compares against a hand-computed expected gain based
    ONLY on the opaque interior region's color."""
    rgb = np.zeros((50, 100, 3), dtype=np.uint8)
    rgb[10:40, 20:80] = 100  # interior "material"
    alpha = np.zeros((50, 100), dtype=np.uint8)
    alpha[10:40, 20:80] = 255

    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    expected_gain = min(max(_DEFAULT_TARGET_MEAN_LUMINANCE / 100, _MIN_GAIN), _MAX_GAIN)
    assert abs(result.applied_gain - expected_gain) < 1e-6


def test_explicit_target_mean_luminance_is_respected(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(100)
    result = lighting_processor.process(
        _make_perspective_result(rgb, alpha), target_mean_luminance=200.0
    )

    assert result.target_mean_luminance == 200.0
    assert result.applied_gain == pytest.approx(2.0)


def test_fully_transparent_material_is_identity_passthrough(
    lighting_processor: LightingProcessor,
) -> None:
    rgb, _ = _solid_material(50)
    alpha = np.zeros((50, 100), dtype=np.uint8)

    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert result.applied_gain == 1.0
    assert np.array_equal(result.adjusted_rgb, rgb)


# --- 8/invalid input handling -----------------------------------------------


def test_wrong_rgb_dimensionality_raises_error(lighting_processor: LightingProcessor) -> None:
    bad_rgb = np.zeros((10, 10), dtype=np.uint8)  # missing channel dim
    alpha = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(LightingAdjustmentError):
        lighting_processor.process(_make_perspective_result(bad_rgb, alpha))


def test_wrong_rgb_dtype_raises_error(lighting_processor: LightingProcessor) -> None:
    bad_rgb = np.zeros((10, 10, 3), dtype=np.float32)
    alpha = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(LightingAdjustmentError):
        lighting_processor.process(_make_perspective_result(bad_rgb, alpha))


def test_empty_rgb_dimension_raises_error(lighting_processor: LightingProcessor) -> None:
    bad_rgb = np.zeros((0, 10, 3), dtype=np.uint8)
    bad_alpha = np.zeros((0, 10), dtype=np.uint8)

    with pytest.raises(LightingAdjustmentError):
        lighting_processor.process(_make_perspective_result(bad_rgb, bad_alpha))


def test_mismatched_alpha_shape_raises_error(lighting_processor: LightingProcessor) -> None:
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    mismatched_alpha = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(LightingAdjustmentError):
        lighting_processor.process(_make_perspective_result(rgb, mismatched_alpha))


def test_target_luminance_out_of_range_raises_error(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(100)

    with pytest.raises(LightingAdjustmentError):
        lighting_processor.process(_make_perspective_result(rgb, alpha), target_mean_luminance=300.0)


def test_negative_target_luminance_raises_error(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(100)

    with pytest.raises(LightingAdjustmentError):
        lighting_processor.process(_make_perspective_result(rgb, alpha), target_mean_luminance=-5.0)


def test_nan_target_luminance_raises_error(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(100)

    with pytest.raises(LightingAdjustmentError):
        lighting_processor.process(
            _make_perspective_result(rgb, alpha), target_mean_luminance=float("nan")
        )


def test_inf_target_luminance_raises_error(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(100)

    with pytest.raises(LightingAdjustmentError):
        lighting_processor.process(
            _make_perspective_result(rgb, alpha), target_mean_luminance=float("inf")
        )


# --- edge cases relevant to the chosen algorithm ----------------------------


def test_boundary_target_values_zero_and_255_are_accepted(
    lighting_processor: LightingProcessor,
) -> None:
    rgb, alpha = _solid_material(100)

    result_zero = lighting_processor.process(
        _make_perspective_result(rgb, alpha), target_mean_luminance=0.0
    )
    result_max = lighting_processor.process(
        _make_perspective_result(rgb, alpha), target_mean_luminance=255.0
    )

    assert result_zero.target_mean_luminance == 0.0
    assert result_max.target_mean_luminance == 255.0


def test_canvas_metadata_is_passed_through_unchanged(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(100)
    perspective_result = _make_perspective_result(rgb, alpha, canvas_offset=(42, 17))

    result = lighting_processor.process(perspective_result)

    assert result.canvas_size == perspective_result.canvas_size
    assert result.canvas_offset == (42, 17)


# --- result schema / immutability ------------------------------------------


def test_result_schema_fields_are_present_and_typed(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(100)
    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert isinstance(result.adjusted_rgb, np.ndarray)
    assert isinstance(result.alpha, np.ndarray)
    assert isinstance(result.canvas_size, tuple)
    assert isinstance(result.canvas_offset, tuple)
    assert isinstance(result.applied_gain, float)
    assert isinstance(result.target_mean_luminance, float)


def test_result_is_frozen(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(100)
    result = lighting_processor.process(_make_perspective_result(rgb, alpha))

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        result.applied_gain = 99.0  # type: ignore[misc]


def test_input_arrays_are_not_mutated(lighting_processor: LightingProcessor) -> None:
    rgb, alpha = _solid_material(30)
    rgb_before = rgb.copy()
    alpha_before = alpha.copy()

    lighting_processor.process(_make_perspective_result(rgb, alpha))

    assert np.array_equal(rgb, rgb_before)
    assert np.array_equal(alpha, alpha_before)
