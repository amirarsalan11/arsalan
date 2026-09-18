"""
Perspective Transformer: computes the homography mapping a material
texture onto the floor-plane quadrilateral described by FloorGeometry,
and warps the material's RGB + alpha into that quadrilateral.

===========================================================================
COORDINATE CONVENTIONS (read before modifying anything in this file)
===========================================================================
- Image coordinate system: origin (0, 0) at the TOP-LEFT of an image;
  x increases RIGHTWARD, y increases DOWNWARD. This matches every
  other stage in this package (numpy array indexing, OpenCV, PIL) and
  is NOT re-derived here — it's inherited directly from geometry.py.
- Corner ordering: (TL, TR, BR, BL) — i.e. top-left, top-right,
  bottom-right, bottom-left, in that exact order. This is
  geometry.py's `GeometryEstimator._approximate_corners` convention
  (top = smaller y, bottom = larger y; left = smaller x, right =
  larger x), reused verbatim here. This module does NOT invent a
  second, independent corner-ordering scheme — `FloorGeometry.corners`
  is trusted to already be in this order, and is only checked for
  well-formedness (count, finiteness, non-degeneracy), never
  re-sorted.
- Source rectangle convention: the material texture's own 4 corners,
  built from `PreparedMaterial.normalized_size = (width, height)` as
  `[(0, 0), (width, 0), (width, height), (0, height)]` — i.e. the same
  (TL, TR, BR, BL) order, so that source point i corresponds to
  destination point i and the homography is not accidentally mirrored
  or rotated.
- Destination quadrilateral convention: `FloorGeometry.corners`,
  shifted into this stage's own LOCAL output canvas (see below), not
  used directly in the original room image's coordinate space.

===========================================================================
WHY A LOCAL, TIGHT CANVAS — NOT THE FULL ROOM IMAGE
===========================================================================
This stage's inputs are exactly `PreparedMaterial` and `FloorGeometry`
(per the milestone's stated goal) — and neither of those carries the
original room image's own width/height. `FloorGeometry.corners` are
absolute pixel coordinates in that (unknown-to-this-stage) room image,
but nothing requires this stage to warp into a canvas of that same
full size: doing so would need a third input this stage doesn't have,
and would waste memory warping into mostly-empty canvas space far from
the floor.

Instead, this stage warps into a TIGHT bounding box around just the
floor corners (plus a small margin for anti-aliased edges), and
records where that box sits within the original room image via
`PerspectiveTransformResult.canvas_offset`. A future Compositing stage
— which DOES have the original room image, e.g. from
`PreprocessedImage.original_image` earlier in the same pipeline run —
pastes this canvas at `canvas_offset` rather than needing Perspective
to have known the full room-image size in advance. This keeps this
stage fully self-contained on exactly its two stated inputs, at the
cost of the future Compositing stage needing to know about the offset
— a small, explicit, documented coupling rather than an implicit one.

===========================================================================
WHY A WARPED IMAGE, NOT JUST A MATRIX
===========================================================================
The milestone's own "Image Transformation" section is specific about
`cv2.warpPerspective`, alpha preservation, interpolation, and border
behavior — none of which would matter if only a matrix were produced.
Both RGB and alpha are warped with the SAME matrix, same canvas size,
same interpolation, and `borderValue=0`, so:
  - RGB and alpha stay pixel-aligned (never computed independently).
  - Alpha is 0 everywhere OUTSIDE the floor quadrilateral, even for a
    source texture that was fully opaque (`has_alpha=False`) — this is
    what lets a future Compositing stage know exactly where the
    material actually landed versus empty canvas margin.
"""

from __future__ import annotations

import cv2
import numpy as np

from ai_engine.exceptions import PerspectiveTransformError
from ai_engine.preprocessing.preprocessor import MAX_SOURCE_DIMENSION
from ai_engine.schemas import FloorGeometry, PerspectiveTransformResult, PreparedMaterial

# Small margin added around the tight bounding box of the floor
# corners, so anti-aliased/interpolated edges from the warp aren't
# hard-clipped exactly at the quadrilateral's boundary.
_CANVAS_MARGIN_PX = 2

# Defensive sanity bound on individual corner coordinate magnitudes —
# catches a corrupted/malformed FloorGeometry before it reaches
# OpenCV, independent of MAX_SOURCE_DIMENSION (a coordinate could be
# absurd without the *canvas* necessarily being oversized, e.g. a
# single wild outlier corner far from the other three).
_MAX_COORDINATE_MAGNITUDE = 100_000.0

# A quadrilateral with (shoelace) area below this many square pixels
# is treated as degenerate (duplicate/collinear corners) — this is an
# absolute, not relative, threshold: any genuinely usable floor
# quadrilateral is many orders of magnitude larger than this.
_MIN_QUADRILATERAL_AREA_PX = 1.0

# Two corners closer together than this are treated as duplicates,
# checked before (and in addition to) the area test for a clearer,
# more specific error message.
_MIN_CORNER_SEPARATION_PX = 1e-6

# A homography matrix with |determinant| below this is treated as
# numerically singular/unusable.
_MIN_HOMOGRAPHY_DETERMINANT = 1e-9

# Reused as the hard ceiling on the OUTPUT canvas's dimensions too —
# same defensive-memory-guard rationale as the Preprocessor and
# Material stages, applied to the bounding box computed here.
_MAX_CANVAS_DIMENSION = MAX_SOURCE_DIMENSION


class PerspectiveTransformer:
    """Stateless — safe to reuse a single instance across requests."""

    def transform(
        self, material: PreparedMaterial, geometry: FloorGeometry
    ) -> PerspectiveTransformResult:
        """Warp `material`'s texture onto the floor quadrilateral in `geometry`.

        Raises:
            PerspectiveTransformError: for any missing/malformed
                geometry corners, degenerate quadrilateral, invalid
                material dimensions, or a numerically unusable
                resulting homography.
        """
        corners = self._validate_corners(geometry.corners)
        self._validate_material(material)

        canvas_size, canvas_offset, local_corners = self._compute_canvas(corners)
        source_points = self._material_rectangle(material.normalized_size)

        homography = self._compute_homography(source_points, local_corners)

        warped_rgb, warped_alpha = self._warp(material, homography, canvas_size)

        destination_corners = tuple(
            (int(round(x)), int(round(y))) for x, y in local_corners.tolist()
        )

        return PerspectiveTransformResult(
            warped_rgb=warped_rgb,
            warped_alpha=warped_alpha,
            homography_matrix=homography,
            canvas_size=canvas_size,
            canvas_offset=canvas_offset,
            destination_corners=destination_corners,
            source_size=material.normalized_size,
            source_has_alpha=material.has_alpha,
        )

    # -- validation --------------------------------------------------------

    def _validate_corners(self, corners: tuple[tuple[int, int], ...]) -> np.ndarray:
        """Defensively validate FloorGeometry.corners: exactly 4
        points, all finite, within a sane magnitude, no duplicates,
        and forming a non-degenerate quadrilateral. Never re-sorts or
        reinterprets the ordering — see module docstring."""
        if corners is None or len(corners) != 4:
            count = 0 if corners is None else len(corners)
            raise PerspectiveTransformError(
                "perspective",
                f"FloorGeometry must have exactly 4 corners; got {count}.",
            )

        try:
            points = np.array(corners, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise PerspectiveTransformError(
                "perspective", "FloorGeometry corners could not be parsed as numeric coordinates.", exc
            ) from exc

        if points.shape != (4, 2):
            raise PerspectiveTransformError(
                "perspective",
                f"FloorGeometry corners must each be an (x, y) pair; got shape {points.shape}.",
            )

        if not np.all(np.isfinite(points)):
            raise PerspectiveTransformError(
                "perspective", "FloorGeometry corners contain non-finite (NaN/Inf) coordinates."
            )

        if np.any(np.abs(points) > _MAX_COORDINATE_MAGNITUDE):
            raise PerspectiveTransformError(
                "perspective",
                f"FloorGeometry corners contain a coordinate exceeding the sane magnitude "
                f"bound of {_MAX_COORDINATE_MAGNITUDE}.",
            )

        self._validate_no_duplicate_corners(points)
        self._validate_nondegenerate_area(points)

        return points

    def _validate_no_duplicate_corners(self, points: np.ndarray) -> None:
        for i in range(4):
            for j in range(i + 1, 4):
                distance = float(np.hypot(*(points[i] - points[j])))
                if distance < _MIN_CORNER_SEPARATION_PX:
                    raise PerspectiveTransformError(
                        "perspective",
                        f"FloorGeometry corners {i} and {j} are duplicate/coincident "
                        f"(distance {distance:.6g}px).",
                    )

    def _validate_nondegenerate_area(self, points: np.ndarray) -> None:
        # Shoelace formula for a simple (non-self-intersecting)
        # quadrilateral, in the given corner order.
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * abs(
            sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4))
        )
        if area < _MIN_QUADRILATERAL_AREA_PX:
            raise PerspectiveTransformError(
                "perspective",
                f"FloorGeometry corners form a degenerate quadrilateral "
                f"(area {area:.6g}px\u00b2, minimum {_MIN_QUADRILATERAL_AREA_PX}).",
            )

    def _validate_material(self, material: PreparedMaterial) -> None:
        width, height = material.normalized_size
        if width <= 0 or height <= 0:
            raise PerspectiveTransformError(
                "perspective",
                f"PreparedMaterial.normalized_size must be positive; got {(width, height)}.",
            )

        if material.texture_rgb.shape[:2] != (height, width):
            raise PerspectiveTransformError(
                "perspective",
                f"PreparedMaterial.texture_rgb shape {material.texture_rgb.shape[:2]} does not "
                f"match normalized_size {(width, height)} (expected (height, width)).",
            )

        if material.alpha_mask.shape[:2] != (height, width):
            raise PerspectiveTransformError(
                "perspective",
                f"PreparedMaterial.alpha_mask shape {material.alpha_mask.shape[:2]} does not "
                f"match normalized_size {(width, height)}.",
            )

    # -- canvas / homography / warp -----------------------------------------

    def _compute_canvas(
        self, corners: np.ndarray
    ) -> tuple[tuple[int, int], tuple[int, int], np.ndarray]:
        """Tight bounding box around `corners`, expanded by a small
        margin, with `corners` re-expressed relative to that box's
        own top-left origin. See module docstring for why this is a
        local canvas rather than the full room image."""
        min_xy = corners.min(axis=0)
        max_xy = corners.max(axis=0)

        offset_x = int(np.floor(min_xy[0])) - _CANVAS_MARGIN_PX
        offset_y = int(np.floor(min_xy[1])) - _CANVAS_MARGIN_PX

        canvas_width = int(np.ceil(max_xy[0] - min_xy[0])) + 2 * _CANVAS_MARGIN_PX
        canvas_height = int(np.ceil(max_xy[1] - min_xy[1])) + 2 * _CANVAS_MARGIN_PX

        canvas_width = max(1, canvas_width)
        canvas_height = max(1, canvas_height)

        if canvas_width > _MAX_CANVAS_DIMENSION or canvas_height > _MAX_CANVAS_DIMENSION:
            raise PerspectiveTransformError(
                "perspective",
                f"Computed canvas size {(canvas_width, canvas_height)} exceeds the maximum "
                f"of {_MAX_CANVAS_DIMENSION}px on a side.",
            )

        local_corners = corners - np.array([offset_x, offset_y], dtype=np.float64)

        return (canvas_width, canvas_height), (offset_x, offset_y), local_corners

    def _material_rectangle(self, normalized_size: tuple[int, int]) -> np.ndarray:
        """The material texture's own 4 corners, in (TL, TR, BR, BL)
        order — see module docstring's "source rectangle convention"."""
        width, height = normalized_size
        return np.array(
            [
                [0.0, 0.0],  # TL
                [float(width), 0.0],  # TR
                [float(width), float(height)],  # BR
                [0.0, float(height)],  # BL
            ],
            dtype=np.float64,
        )

    def _compute_homography(self, source_points: np.ndarray, destination_points: np.ndarray) -> np.ndarray:
        try:
            homography = cv2.getPerspectiveTransform(
                source_points.astype(np.float32), destination_points.astype(np.float32)
            )
        except cv2.error as exc:
            raise PerspectiveTransformError(
                "perspective", "OpenCV failed to compute the perspective transform.", exc
            ) from exc

        if homography is None or not np.all(np.isfinite(homography)):
            raise PerspectiveTransformError(
                "perspective", "Computed homography matrix contains non-finite values."
            )

        determinant = float(np.linalg.det(homography))
        if abs(determinant) < _MIN_HOMOGRAPHY_DETERMINANT:
            raise PerspectiveTransformError(
                "perspective",
                f"Computed homography matrix is numerically singular "
                f"(|determinant|={abs(determinant):.3g}).",
            )

        return homography.astype(np.float64)

    def _warp(
        self,
        material: PreparedMaterial,
        homography: np.ndarray,
        canvas_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Warp RGB and alpha with the SAME matrix/canvas/interpolation
        so they stay pixel-aligned; borderValue=0 on both marks
        everything outside the floor quad as empty/transparent."""
        homography_f32 = homography.astype(np.float32)

        warped_rgb = cv2.warpPerspective(
            material.texture_rgb,
            homography_f32,
            canvas_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_alpha = cv2.warpPerspective(
            material.alpha_mask,
            homography_f32,
            canvas_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return warped_rgb, warped_alpha
