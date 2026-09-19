"""
Geometry Estimator: contour extraction, Hough line detection,
vanishing-point estimation via robust voting, and floor-plane corner
approximation, with graceful degradation when the vanishing point
can't be reliably resolved.

See ai_engine/schemas.py:FloorGeometry for the output shape.
"""

from __future__ import annotations

import itertools
import math

import cv2
import numpy as np

from ai_engine.exceptions import GeometryEstimationError
from ai_engine.schemas import FloorGeometry, LineSegment

# Polygon simplification: epsilon as a fraction of contour perimeter.
_POLYGON_SIMPLIFICATION_EPSILON_FRACTION = 0.01

# Canny edge detection ROI: dilate the mask boundary by this fraction
# of the image diagonal to focus the search on the floor boundary
# region, rather than picking up unrelated lines elsewhere in the room.
_ROI_DILATION_DIAGONAL_FRACTION = 0.03

# Hough transform parameters, expressed relative to image diagonal for
# resolution independence.
_HOUGH_MIN_LINE_LENGTH_FRACTION = 0.05
_HOUGH_MAX_LINE_GAP_FRACTION = 0.02
_HOUGH_THRESHOLD = 40
_HOUGH_RHO = 1
_HOUGH_THETA = math.pi / 180

# A vanishing point requires at least this many corroborating line
# pairs voting for the same cluster before it's trusted at all.
_MIN_VANISHING_POINT_VOTES = 3

# Two lines with an angle difference below this are treated as
# near-parallel and skipped for intersection (the 2x2 linear system
# would be ill-conditioned).
_MIN_ANGLE_DIFFERENCE_DEGREES = 10.0

# Candidate intersections farther than this multiple of the image
# diagonal from the image center are discarded as numerically
# unstable / meaningless outliers.
_MAX_INTERSECTION_DISTANCE_DIAGONALS = 3.0

# Radius (as a fraction of image diagonal) used to cluster candidate
# vanishing-point intersections during voting.
_VOTING_CLUSTER_RADIUS_DIAGONAL_FRACTION = 0.05

# Confidence assigned when geometry degrades to polygon-only (no
# vanishing point resolved), vs. the higher confidence when a
# vanishing point was found.
_CONFIDENCE_WITH_VANISHING_POINT = 0.9
_CONFIDENCE_POLYGON_ONLY = 0.5


class GeometryEstimator:
    """Stateless — safe to reuse a single instance across requests."""

    def estimate(self, binary_mask: np.ndarray, grayscale_image: np.ndarray) -> FloorGeometry:
        """
        Args:
            binary_mask: original-resolution uint8 {0,255} floor mask
                (ProcessedMask.binary_mask).
            grayscale_image: original-resolution single-channel image,
                used for line detection (real-world lines like
                plank/grout seams are far better represented in the
                actual image than in the segmentation mask alone).

        Raises:
            GeometryEstimationError: if even a degraded, polygon-only
                estimate cannot be produced (e.g. empty/degenerate mask).
        """
        contour = self._extract_largest_contour(binary_mask)
        simplified_polygon = self._simplify_polygon(contour)
        corners = self._approximate_corners(simplified_polygon)

        lines = self._detect_lines(binary_mask, grayscale_image)
        vanishing_point = self._estimate_vanishing_point(lines, grayscale_image.shape)

        confidence = (
            _CONFIDENCE_WITH_VANISHING_POINT
            if vanishing_point is not None
            else _CONFIDENCE_POLYGON_ONLY
        )

        return FloorGeometry(
            contour=contour,
            simplified_polygon=simplified_polygon,
            corners=corners,
            lines=lines,
            vanishing_point=vanishing_point,
            geometry_confidence=confidence,
        )

    # -- contour / polygon / corners --------------------------------------

    def _extract_largest_contour(self, binary_mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise GeometryEstimationError(
                "geometry", "No contour could be extracted from the floor mask."
            )

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) <= 0:
            raise GeometryEstimationError(
                "geometry", "The largest contour found has zero area."
            )

        return largest.reshape(-1, 2)

    def _simplify_polygon(self, contour: np.ndarray) -> np.ndarray:
        perimeter = cv2.arcLength(contour.reshape(-1, 1, 2), closed=True)
        epsilon = max(1.0, perimeter * _POLYGON_SIMPLIFICATION_EPSILON_FRACTION)
        simplified = cv2.approxPolyDP(contour.reshape(-1, 1, 2), epsilon, closed=True)
        return simplified.reshape(-1, 2)

    def _approximate_corners(self, polygon: np.ndarray) -> tuple[tuple[int, int], ...]:
        """Pick four extremal corners from the simplified polygon,
        ordered (TL, TR, BR, BL) — the shape a future Perspective
        Engine milestone would use as homography source points."""
        if polygon.shape[0] < 3:
            raise GeometryEstimationError(
                "geometry",
                f"Simplified polygon has only {polygon.shape[0]} points; "
                "at least 3 are required to approximate corners.",
            )

        points = polygon.astype(np.float64)
        sums = points[:, 0] + points[:, 1]
        diffs = points[:, 0] - points[:, 1]

        top_left = points[np.argmin(sums)]
        bottom_right = points[np.argmax(sums)]
        top_right = points[np.argmax(diffs)]
        bottom_left = points[np.argmin(diffs)]

        ordered = [top_left, top_right, bottom_right, bottom_left]
        return tuple((int(round(p[0])), int(round(p[1]))) for p in ordered)

    # -- line detection -----------------------------------------------------

    def _detect_lines(self, binary_mask: np.ndarray, grayscale_image: np.ndarray) -> list[LineSegment]:
        height, width = grayscale_image.shape[:2]
        diagonal = float(np.hypot(width, height))

        roi = self._build_roi(binary_mask, diagonal)

        # Adaptive Canny thresholds via Otsu's method on the ROI's
        # gradient magnitude, rather than fixed constants — so the
        # same code handles both high- and low-contrast scenes
        # reasonably instead of being tuned to one lighting scenario.
        masked_gray = cv2.bitwise_and(grayscale_image, grayscale_image, mask=roi)
        otsu_threshold, _ = cv2.threshold(
            masked_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        low_threshold = max(1.0, otsu_threshold * 0.5)
        high_threshold = max(low_threshold + 1.0, otsu_threshold * 1.5)

        edges = cv2.Canny(masked_gray, low_threshold, high_threshold)

        min_line_length = max(10, round(diagonal * _HOUGH_MIN_LINE_LENGTH_FRACTION))
        max_line_gap = max(1, round(diagonal * _HOUGH_MAX_LINE_GAP_FRACTION))

        raw_lines = cv2.HoughLinesP(
            edges,
            _HOUGH_RHO,
            _HOUGH_THETA,
            _HOUGH_THRESHOLD,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        if raw_lines is None:
            return []

        segments = []
        for line in raw_lines:
            x1, y1, x2, y2 = (int(v) for v in line[0])
            angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
            segments.append(LineSegment(x1=x1, y1=y1, x2=x2, y2=y2, angle_deg=angle_deg))

        return segments

    def _build_roi(self, binary_mask: np.ndarray, diagonal: float) -> np.ndarray:
        """Dilate the floor mask boundary to focus edge/line detection
        near the floor, rather than the whole frame."""
        dilation_size = max(3, round(diagonal * _ROI_DILATION_DIAGONAL_FRACTION))
        if dilation_size % 2 == 0:
            dilation_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
        return cv2.dilate(binary_mask, kernel)

    # -- vanishing point ------------------------------------------------------

    def _estimate_vanishing_point(
        self, lines: list[LineSegment], image_shape: tuple[int, ...]
    ) -> tuple[float, float] | None:
        if len(lines) < 2:
            return None

        height, width = image_shape[:2]
        diagonal = float(np.hypot(width, height))
        center = (width / 2.0, height / 2.0)
        max_distance = diagonal * _MAX_INTERSECTION_DISTANCE_DIAGONALS

        candidates: list[tuple[float, float]] = []
        for line_a, line_b in itertools.combinations(lines, 2):
            angle_diff = abs(line_a.angle_deg - line_b.angle_deg)
            angle_diff = min(angle_diff, 180.0 - angle_diff)
            if angle_diff < _MIN_ANGLE_DIFFERENCE_DEGREES:
                # Near-parallel lines: the 2x2 system would be
                # ill-conditioned, and any "intersection" found would
                # be numerically unstable rather than meaningful.
                continue

            intersection = self._line_intersection(line_a, line_b)
            if intersection is None:
                continue

            ix, iy = intersection
            distance_from_center = math.hypot(ix - center[0], iy - center[1])
            if distance_from_center > max_distance:
                # Degenerate/outlier intersection, e.g. from two lines
                # that are almost-but-not-quite parallel.
                continue

            candidates.append((ix, iy))

        if not candidates:
            return None

        return self._cluster_and_vote(candidates, diagonal)

    def _line_intersection(
        self, line_a: LineSegment, line_b: LineSegment
    ) -> tuple[float, float] | None:
        """Solve the 2x2 linear system for the intersection of two
        infinite lines extended from each segment, in general form
        a*x + b*y = c."""
        a1 = line_a.y2 - line_a.y1
        b1 = line_a.x1 - line_a.x2
        c1 = a1 * line_a.x1 + b1 * line_a.y1

        a2 = line_b.y2 - line_b.y1
        b2 = line_b.x1 - line_b.x2
        c2 = a2 * line_b.x1 + b2 * line_b.y1

        determinant = a1 * b2 - a2 * b1
        if abs(determinant) < 1e-9:
            return None  # parallel or coincident lines

        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return (x, y)

    def _cluster_and_vote(
        self, candidates: list[tuple[float, float]], diagonal: float
    ) -> tuple[float, float] | None:
        """Radius-based clustering: for each candidate, count how many
        other candidates fall within the voting radius; the candidate
        with the most support defines the winning cluster, whose
        centroid is the final vanishing point estimate. Rejects the
        result entirely if no cluster reaches the minimum vote count —
        callers should treat a None return as "could not resolve",
        not as "vanishing point is at the origin"."""
        radius = diagonal * _VOTING_CLUSTER_RADIUS_DIAGONAL_FRACTION
        points = np.array(candidates, dtype=np.float64)

        best_cluster_indices: np.ndarray | None = None
        best_count = 0

        for i, point in enumerate(points):
            distances = np.hypot(points[:, 0] - point[0], points[:, 1] - point[1])
            within_radius = distances <= radius
            count = int(np.count_nonzero(within_radius))
            if count > best_count:
                best_count = count
                best_cluster_indices = within_radius

        if best_cluster_indices is None or best_count < _MIN_VANISHING_POINT_VOTES:
            return None

        cluster_points = points[best_cluster_indices]
        centroid = cluster_points.mean(axis=0)
        return (float(centroid[0]), float(centroid[1]))
