"""
Tests for perspective/perspective_transformer.py.

All synthetic, in-memory PreparedMaterial/FloorGeometry objects — no
model weights, external files, or network access needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_engine.exceptions import PerspectiveTransformError
from ai_engine.perspective.perspective_transformer import (
    _MAX_COORDINATE_MAGNITUDE,
    PerspectiveTransformer,
)
from ai_engine.schemas import FloorGeometry, PerspectiveTransformResult, PreparedMaterial


@pytest.fixture
def transformer() -> PerspectiveTransformer:
    return PerspectiveTransformer()


def _make_material(
    width: int = 100,
    height: int = 50,
    color: int = 120,
    has_alpha: bool = False,
    alpha_value: int = 255,
) -> PreparedMaterial:
    return PreparedMaterial(
        texture_rgb=np.full((height, width, 3), color, dtype=np.uint8),
        alpha_mask=np.full((height, width), alpha_value, dtype=np.uint8),
        has_alpha=has_alpha,
        original_size=(width, height),
        normalized_size=(width, height),
        source_format="PNG",
        was_downscaled=False,
    )


def _make_geometry(corners: tuple, confidence: float = 0.5) -> FloorGeometry:
    return FloorGeometry(
        contour=np.array([[0, 0]]),
        simplified_polygon=np.array([[0, 0]]),
        corners=corners,
        lines=[],
        vanishing_point=None,
        geometry_confidence=confidence,
    )


_TRAPEZOID_CORNERS = ((220, 200), (420, 200), (540, 460), (100, 460))  # TL, TR, BR, BL


# --- 1/2/3/4: valid transformation, geometry integration, matrix, dims -----


def test_valid_rectangle_to_quadrilateral_transformation(transformer: PerspectiveTransformer) -> None:
    material = _make_material()
    geometry = _make_geometry(_TRAPEZOID_CORNERS)

    result = transformer.transform(material, geometry)

    assert isinstance(result, PerspectiveTransformResult)
    assert result.warped_rgb.shape[:2] == (result.canvas_size[1], result.canvas_size[0])


def test_valid_floor_geometry_integration_uses_real_corners(
    transformer: PerspectiveTransformer,
) -> None:
    material = _make_material()
    geometry = _make_geometry(_TRAPEZOID_CORNERS)

    result = transformer.transform(material, geometry)

    # destination_corners must be the SAME corners, just shifted into
    # local canvas space by exactly -canvas_offset.
    offset_x, offset_y = result.canvas_offset
    for (gx, gy), (dx, dy) in zip(_TRAPEZOID_CORNERS, result.destination_corners):
        assert dx == gx - offset_x
        assert dy == gy - offset_y


def test_homography_matrix_is_3x3_and_finite(transformer: PerspectiveTransformer) -> None:
    material = _make_material()
    geometry = _make_geometry(_TRAPEZOID_CORNERS)

    result = transformer.transform(material, geometry)

    assert result.homography_matrix.shape == (3, 3)
    assert np.all(np.isfinite(result.homography_matrix))


def test_output_dimensions_match_canvas_size(transformer: PerspectiveTransformer) -> None:
    material = _make_material()
    geometry = _make_geometry(_TRAPEZOID_CORNERS)

    result = transformer.transform(material, geometry)

    width, height = result.canvas_size
    assert result.warped_rgb.shape == (height, width, 3)
    assert result.warped_alpha.shape == (height, width)


# --- 5: corner ordering ----------------------------------------------------


def test_corner_ordering_is_preserved_not_resorted(transformer: PerspectiveTransformer) -> None:
    """Deliberately non-axis-aligned corners in a valid TL,TR,BR,BL
    order — confirms the transformer trusts the given order rather
    than re-deriving it independently."""
    material = _make_material()
    corners = ((50, 40), (150, 45), (145, 140), (45, 135))  # a slightly skewed quad
    geometry = _make_geometry(corners)

    result = transformer.transform(material, geometry)
    offset_x, offset_y = result.canvas_offset

    for (gx, gy), (dx, dy) in zip(corners, result.destination_corners):
        assert dx == gx - offset_x
        assert dy == gy - offset_y


# --- 6/7: identity-like and non-trivial perspective ------------------------


def test_identity_like_transformation_preserves_texture_values(
    transformer: PerspectiveTransformer,
) -> None:
    """Floor quad exactly matching the material's own rectangle shape
    (just translated) should produce a warp that's effectively a pure
    translation — sampled interior pixels should closely match the
    original texture color."""
    material = _make_material(width=100, height=50, color=120)
    geometry = _make_geometry(((10, 10), (110, 10), (110, 60), (10, 60)))

    result = transformer.transform(material, geometry)

    cy, cx = result.canvas_size[1] // 2, result.canvas_size[0] // 2
    assert abs(int(result.warped_rgb[cy, cx, 0]) - 120) < 5


def test_non_trivial_perspective_distortion_is_applied(transformer: PerspectiveTransformer) -> None:
    """A genuinely trapezoidal (non-rectangular) destination must
    produce a homography that is NOT a pure similarity transform —
    confirmed by the matrix having non-trivial perspective terms."""
    material = _make_material()
    geometry = _make_geometry(_TRAPEZOID_CORNERS)

    result = transformer.transform(material, geometry)

    # The bottom row of a pure affine transform is always [0, 0, 1].
    # A genuine perspective warp (converging trapezoid) has non-zero
    # values in H[2,0] or H[2,1].
    bottom_row = result.homography_matrix[2, :2]
    assert np.any(np.abs(bottom_row) > 1e-6)


# --- 8/9: alpha preservation and spatial alignment --------------------------


def test_alpha_is_preserved_inside_quad_and_zero_outside(
    transformer: PerspectiveTransformer,
) -> None:
    material = _make_material(has_alpha=True, alpha_value=200)
    geometry = _make_geometry(_TRAPEZOID_CORNERS)

    result = transformer.transform(material, geometry)

    assert result.warped_alpha[0, 0] == 0  # canvas corner, outside the quad
    cy, cx = result.canvas_size[1] // 2, result.canvas_size[0] // 2
    assert result.warped_alpha[cy, cx] > 150  # inside the quad, near-original alpha


def test_partial_transparency_is_not_flattened_to_opaque(transformer: PerspectiveTransformer) -> None:
    material = _make_material(has_alpha=True, alpha_value=128)
    geometry = _make_geometry(((10, 10), (110, 10), (110, 60), (10, 60)))

    result = transformer.transform(material, geometry)

    cy, cx = result.canvas_size[1] // 2, result.canvas_size[0] // 2
    # Should stay close to the source's 128, not be pushed to 0 or 255.
    assert 100 < result.warped_alpha[cy, cx] < 160


def test_rgb_and_alpha_stay_spatially_aligned(transformer: PerspectiveTransformer) -> None:
    material = _make_material()
    geometry = _make_geometry(_TRAPEZOID_CORNERS)

    result = transformer.transform(material, geometry)

    assert result.warped_rgb.shape[:2] == result.warped_alpha.shape


def test_source_has_alpha_is_passed_through(transformer: PerspectiveTransformer) -> None:
    opaque_material = _make_material(has_alpha=False)
    transparent_material = _make_material(has_alpha=True)
    geometry = _make_geometry(_TRAPEZOID_CORNERS)

    opaque_result = transformer.transform(opaque_material, geometry)
    transparent_result = transformer.transform(transparent_material, geometry)

    assert opaque_result.source_has_alpha is False
    assert transparent_result.source_has_alpha is True


# --- 10/11/12/13/14: invalid geometry -----------------------------------


def test_missing_corners_raises_perspective_transform_error(
    transformer: PerspectiveTransformer,
) -> None:
    material = _make_material()
    geometry = _make_geometry(())

    with pytest.raises(PerspectiveTransformError):
        transformer.transform(material, geometry)


def test_wrong_corner_count_raises_perspective_transform_error(
    transformer: PerspectiveTransformer,
) -> None:
    material = _make_material()
    geometry = _make_geometry(((0, 0), (10, 0), (10, 10)))  # only 3

    with pytest.raises(PerspectiveTransformError):
        transformer.transform(material, geometry)


def test_too_many_corners_raises_perspective_transform_error(
    transformer: PerspectiveTransformer,
) -> None:
    material = _make_material()
    geometry = _make_geometry(((0, 0), (10, 0), (10, 10), (0, 10), (5, 5)))  # 5

    with pytest.raises(PerspectiveTransformError):
        transformer.transform(material, geometry)


def test_duplicate_corners_raise_perspective_transform_error(
    transformer: PerspectiveTransformer,
) -> None:
    material = _make_material()
    geometry = _make_geometry(((0, 0), (0, 0), (10, 10), (0, 10)))

    with pytest.raises(PerspectiveTransformError):
        transformer.transform(material, geometry)


def test_collinear_degenerate_corners_raise_perspective_transform_error(
    transformer: PerspectiveTransformer,
) -> None:
    material = _make_material()
    geometry = _make_geometry(((0, 0), (10, 0), (20, 0), (30, 0)))  # all on one line

    with pytest.raises(PerspectiveTransformError):
        transformer.transform(material, geometry)


def test_nan_coordinate_raises_perspective_transform_error(
    transformer: PerspectiveTransformer,
) -> None:
    material = _make_material()
    geometry = _make_geometry(((0, 0), (10, 0), (float("nan"), 10), (0, 10)))

    with pytest.raises(PerspectiveTransformError):
        transformer.transform(material, geometry)


def test_inf_coordinate_raises_perspective_transform_error(
    transformer: PerspectiveTransformer,
) -> None:
    material = _make_material()
    geometry = _make_geometry(((0, 0), (10, 0), (float("inf"), 10), (0, 10)))

    with pytest.raises(PerspectiveTransformError):
        transformer.transform(material, geometry)


def test_coordinate_exceeding_sane_magnitude_raises_error(
    transformer: PerspectiveTransformer,
) -> None:
    material = _make_material()
    huge = _MAX_COORDINATE_MAGNITUDE * 10
    geometry = _make_geometry(((0, 0), (10, 0), (huge, 10), (0, 10)))

    with pytest.raises(PerspectiveTransformError):
        transformer.transform(material, geometry)


# --- 15: invalid material dimensions -----------------------------------


def test_zero_dimension_material_raises_error(transformer: PerspectiveTransformer) -> None:
    material = PreparedMaterial(
        texture_rgb=np.zeros((0, 0, 3), dtype=np.uint8),
        alpha_mask=np.zeros((0, 0), dtype=np.uint8),
        has_alpha=False,
        original_size=(0, 0),
        normalized_size=(0, 0),
        source_format="PNG",
        was_downscaled=False,
    )
    geometry = _make_geometry(_TRAPEZOID_CORNERS)

    with pytest.raises(PerspectiveTransformError):
        transformer.transform(material, geometry)


def test_material_shape_mismatch_raises_error(transformer: PerspectiveTransformer) -> None:
    material = PreparedMaterial(
        texture_rgb=np.full((50, 100, 3), 100, dtype=np.uint8),
        alpha_mask=np.full((50, 100), 255, dtype=np.uint8),
        has_alpha=False,
        original_size=(100, 50),
        normalized_size=(999, 999),  # deliberately mismatched vs actual array shape
        source_format="PNG",
        was_downscaled=False,
    )
    geometry = _make_geometry(_TRAPEZOID_CORNERS)

    with pytest.raises(PerspectiveTransformError):
        transformer.transform(material, geometry)


# --- 16: deterministic repeated execution -----------------------------------


def test_transform_is_deterministic_across_repeated_calls(
    transformer: PerspectiveTransformer,
) -> None:
    material = _make_material()
    geometry = _make_geometry(_TRAPEZOID_CORNERS)

    result_a = transformer.transform(material, geometry)
    result_b = transformer.transform(material, geometry)

    assert np.array_equal(result_a.warped_rgb, result_b.warped_rgb)
    assert np.array_equal(result_a.warped_alpha, result_b.warped_alpha)
    assert np.array_equal(result_a.homography_matrix, result_b.homography_matrix)
    assert result_a.canvas_size == result_b.canvas_size
    assert result_a.canvas_offset == result_b.canvas_offset


# --- 17: invalid transformation detection (singular matrix) -----------------


def test_near_singular_homography_is_detected() -> None:
    """Directly exercise the private homography validator against a
    deliberately singular matrix, rather than trying to coax
    getPerspectiveTransform into producing one indirectly."""
    transformer = PerspectiveTransformer()
    singular_source = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    # Collapse all destination points onto a single line -> singular.
    singular_destination = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    with pytest.raises(PerspectiveTransformError):
        transformer._compute_homography(singular_source, singular_destination)


# --- 18: extreme but valid geometry -----------------------------------------


def test_extreme_thin_but_valid_quadrilateral_does_not_raise(
    transformer: PerspectiveTransformer,
) -> None:
    material = _make_material()
    # A thin sliver quad, still non-degenerate (area well above the minimum).
    geometry = _make_geometry(((0, 0), (1000, 0), (1000, 10), (0, 50)))

    result = transformer.transform(material, geometry)
    assert np.all(np.isfinite(result.homography_matrix))


def test_extreme_large_quadrilateral_within_bounds(transformer: PerspectiveTransformer) -> None:
    material = _make_material()
    geometry = _make_geometry(((0, 0), (5000, 0), (5000, 5000), (0, 5000)))

    result = transformer.transform(material, geometry)
    assert max(result.canvas_size) <= 5000 + 10


def test_oversized_canvas_raises_error(transformer: PerspectiveTransformer) -> None:
    material = _make_material()
    geometry = _make_geometry(((0, 0), (50_000, 0), (50_000, 50_000), (0, 50_000)))

    with pytest.raises(PerspectiveTransformError):
        transformer.transform(material, geometry)


# --- 19: output schema validation ------------------------------------------


def test_output_schema_fields_are_present_and_typed(transformer: PerspectiveTransformer) -> None:
    material = _make_material()
    geometry = _make_geometry(_TRAPEZOID_CORNERS)

    result = transformer.transform(material, geometry)

    assert isinstance(result.warped_rgb, np.ndarray)
    assert isinstance(result.warped_alpha, np.ndarray)
    assert isinstance(result.homography_matrix, np.ndarray)
    assert isinstance(result.canvas_size, tuple)
    assert isinstance(result.canvas_offset, tuple)
    assert isinstance(result.destination_corners, tuple)
    assert len(result.destination_corners) == 4
    assert isinstance(result.source_size, tuple)
    assert isinstance(result.source_has_alpha, bool)


def test_output_is_frozen(transformer: PerspectiveTransformer) -> None:
    material = _make_material()
    geometry = _make_geometry(_TRAPEZOID_CORNERS)
    result = transformer.transform(material, geometry)

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        result.canvas_size = (1, 1)  # type: ignore[misc]


# --- 20: no unintended modification of input arrays -------------------------


def test_material_arrays_are_not_mutated(transformer: PerspectiveTransformer) -> None:
    material = _make_material()
    rgb_before = material.texture_rgb.copy()
    alpha_before = material.alpha_mask.copy()
    geometry = _make_geometry(_TRAPEZOID_CORNERS)

    transformer.transform(material, geometry)

    assert np.array_equal(material.texture_rgb, rgb_before)
    assert np.array_equal(material.alpha_mask, alpha_before)


def test_geometry_corners_are_not_mutated(transformer: PerspectiveTransformer) -> None:
    material = _make_material()
    corners_before = _TRAPEZOID_CORNERS
    geometry = _make_geometry(_TRAPEZOID_CORNERS)

    transformer.transform(material, geometry)

    assert geometry.corners == corners_before
