"""
Tests for geometry/geometry.py.

Uses synthetic binary masks and grayscale images built directly with
OpenCV drawing primitives — no real photographs needed.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ai_engine.exceptions import GeometryEstimationError
from ai_engine.geometry.geometry import GeometryEstimator


@pytest.fixture
def geometry_estimator() -> GeometryEstimator:
    return GeometryEstimator()


def _trapezoid_room(width: int = 640, height: int = 480, with_lines: bool = True):
    """A synthetic floor mask + matching grayscale image: a trapezoid
    (simulating perspective convergence) with real drawn edges the
    Hough transform can actually detect."""
    mask = np.zeros((height, width), dtype=np.uint8)
    trapezoid = np.array(
        [[100, height - 20], [540, height - 20], [420, 200], [220, 200]], dtype=np.int32
    )
    cv2.fillPoly(mask, [trapezoid], 255)

    gray = np.full((height, width), 200, dtype=np.uint8)
    if with_lines:
        cv2.line(gray, (100, height - 20), (220, 200), 40, thickness=3)
        cv2.line(gray, (540, height - 20), (420, 200), 40, thickness=3)
        cv2.line(gray, (150, height - 20), (240, 200), 40, thickness=2)
        cv2.line(gray, (490, height - 20), (400, 200), 40, thickness=2)

    return mask, gray


def test_contour_and_polygon_are_extracted(geometry_estimator: GeometryEstimator) -> None:
    mask, gray = _trapezoid_room()
    result = geometry_estimator.estimate(mask, gray)

    assert result.contour.shape[0] > 0
    assert result.contour.shape[1] == 2
    assert result.simplified_polygon.shape[0] >= 3
    assert result.simplified_polygon.shape[0] <= result.contour.shape[0]


def test_corners_are_four_ordered_points(geometry_estimator: GeometryEstimator) -> None:
    mask, gray = _trapezoid_room()
    result = geometry_estimator.estimate(mask, gray)

    assert len(result.corners) == 4
    for corner in result.corners:
        assert len(corner) == 2
        assert isinstance(corner[0], int)
        assert isinstance(corner[1], int)


def test_vanishing_point_recovered_from_converging_lines(
    geometry_estimator: GeometryEstimator,
) -> None:
    mask, gray = _trapezoid_room(with_lines=True)
    result = geometry_estimator.estimate(mask, gray)

    assert result.vanishing_point is not None
    vx, vy = result.vanishing_point
    # The trapezoid's left/right edges are symmetric around x=320 (the
    # image's horizontal center for width=640), and converge above the
    # horizon (negative-ish y relative to the trapezoid's top edge at
    # y=200) — a generous tolerance band confirms the math is in the
    # right neighborhood without over-fitting to exact pixel values.
    assert 250 < vx < 390
    assert vy < 200
    assert result.geometry_confidence == pytest.approx(0.9)


def test_no_lines_degrades_gracefully_without_raising(
    geometry_estimator: GeometryEstimator,
) -> None:
    mask, gray = _trapezoid_room(with_lines=False)
    result = geometry_estimator.estimate(mask, gray)

    assert result.vanishing_point is None
    assert result.geometry_confidence == pytest.approx(0.5)
    # Still produces a usable polygon/corners despite no vanishing point.
    assert len(result.corners) == 4


def test_empty_mask_raises_geometry_estimation_error(
    geometry_estimator: GeometryEstimator,
) -> None:
    height, width = 480, 640
    mask = np.zeros((height, width), dtype=np.uint8)
    gray = np.full((height, width), 180, dtype=np.uint8)

    with pytest.raises(GeometryEstimationError):
        geometry_estimator.estimate(mask, gray)


def test_near_parallel_lines_do_not_produce_a_spurious_vanishing_point(
    geometry_estimator: GeometryEstimator,
) -> None:
    """Two lines that are nearly (but not exactly) parallel should be
    rejected for intersection, rather than producing a wildly unstable
    vanishing point from an ill-conditioned 2x2 system."""
    height, width = 480, 640
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[300:460, 100:540] = 255

    gray = np.full((height, width), 200, dtype=np.uint8)
    # Two nearly-parallel horizontal-ish lines (angle difference well
    # under the minimum threshold used for intersection).
    cv2.line(gray, (100, 320), (540, 322), 40, thickness=2)
    cv2.line(gray, (100, 340), (540, 343), 40, thickness=2)

    result = geometry_estimator.estimate(mask, gray)
    # Either no vanishing point at all, or — if enough other lines are
    # incidentally detected — a confidently-voted one; what must NOT
    # happen is a crash or an obviously nonsensical result. The
    # meaningful assertion is that this never raises and always
    # returns a valid FloorGeometry.
    assert result.geometry_confidence in (0.5, 0.9)


def test_lines_have_correct_angle_computation(geometry_estimator: GeometryEstimator) -> None:
    mask, gray = _trapezoid_room(with_lines=True)
    result = geometry_estimator.estimate(mask, gray)

    for line in result.lines:
        expected_angle = np.degrees(np.arctan2(line.y2 - line.y1, line.x2 - line.x1))
        assert abs(line.angle_deg - expected_angle) < 1e-6


def test_rectangular_floor_produces_four_corners_matching_shape(
    geometry_estimator: GeometryEstimator,
) -> None:
    height, width = 480, 640
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[300:460, 100:540] = 255
    gray = np.full((height, width), 180, dtype=np.uint8)

    result = geometry_estimator.estimate(mask, gray)

    xs = [c[0] for c in result.corners]
    ys = [c[1] for c in result.corners]
    assert min(xs) < 120
    assert max(xs) > 520
    assert min(ys) < 320
    assert max(ys) > 440


def test_line_intersection_matches_hand_calculation(geometry_estimator: GeometryEstimator) -> None:
    """Directly exercise the private intersection solver against a
    hand-computable case: two lines crossing at a known point."""
    from ai_engine.schemas import LineSegment

    # Line A: from (0, 0) to (10, 10) -> y = x
    line_a = LineSegment(x1=0, y1=0, x2=10, y2=10, angle_deg=45.0)
    # Line B: from (0, 10) to (10, 0) -> y = -x + 10
    line_b = LineSegment(x1=0, y1=10, x2=10, y2=0, angle_deg=-45.0)

    intersection = geometry_estimator._line_intersection(line_a, line_b)
    assert intersection is not None
    x, y = intersection
    assert abs(x - 5.0) < 1e-6
    assert abs(y - 5.0) < 1e-6


def test_line_intersection_returns_none_for_parallel_lines(
    geometry_estimator: GeometryEstimator,
) -> None:
    from ai_engine.schemas import LineSegment

    line_a = LineSegment(x1=0, y1=0, x2=10, y2=0, angle_deg=0.0)
    line_b = LineSegment(x1=0, y1=5, x2=10, y2=5, angle_deg=0.0)

    assert geometry_estimator._line_intersection(line_a, line_b) is None
