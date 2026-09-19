"""
Compositor: combines the room's own image with the lighting-adjusted,
perspective-warped material, restricted to the intersection of the
material's own alpha and the actual floor segmentation mask, producing
a final render at the room image's own original resolution.

===========================================================================
WHY THE FLOOR MASK IS INTERSECTED IN, NOT JUST THE MATERIAL'S ALPHA
===========================================================================
PerspectiveTransformResult/LightingResult's `alpha` is 0 outside the
floor QUADRILATERAL (the 4-corner approximation Geometry produced) and
follows the material's own shape inside it — but a real floor is
rarely a perfect quadrilateral, and the quad's bounding region may
still cover furniture, rugs, or other objects sitting on the floor
that were correctly excluded from the original floor mask
(ProcessedMask.binary_mask) back in Mask Processing. Using the
material alpha alone would paint material over those obstructions.
This stage intersects the two: a pixel is only replaced if BOTH the
warped material has non-zero alpha there AND the true floor mask says
that pixel is floor. `ProcessedMask.binary_mask` is treated as the
authoritative floor shape; the perspective canvas's alpha is treated
as "the material's own footprint within whatever region is passed to
it," never as a substitute for the real segmentation result.

===========================================================================
COORDINATE HANDLING
===========================================================================
`LightingResult.canvas_offset` (inherited unchanged from
PerspectiveTransformResult) can legitimately place the canvas partly
or, in a degenerate/malformed-input case, entirely outside the room
image's own bounds (Perspective's tight-bounding-box canvas includes
a small margin that could push it slightly negative, or a test/caller
could construct an out-of-range offset directly). This stage computes
the actual pixel intersection between the canvas and the room image
and operates only on that overlap — never raises for a partial or
even fully-empty overlap; a fully-empty overlap simply returns the
room image unchanged (`applied_pixel_count=0`), since there is nothing
graceful to paint in that case, but that is not an error.

===========================================================================
BLENDING
===========================================================================
Standard alpha compositing: `output = room * (1 - a) + material * a`,
where `a` is the intersected alpha normalized to [0, 1]. This is a
deterministic, bounded, per-pixel convex combination — never a
learned/artistic operation, and mathematically guaranteed to stay
within the valid uint8 range without needing a corrective clip
(though one is applied anyway as a defensive safety net against
floating-point rounding at the exact 0/255 boundaries).
"""

from __future__ import annotations

import numpy as np

from ai_engine.exceptions import CompositingError
from ai_engine.schemas import CompositingResult, LightingResult, PreprocessedImage, ProcessedMask


class Compositor:
    """Stateless — safe to reuse a single instance across requests."""

    def composite(
        self,
        preprocessed: PreprocessedImage,
        mask: ProcessedMask,
        lighting_result: LightingResult,
    ) -> CompositingResult:
        """Paste `lighting_result`'s lit material into `preprocessed`'s
        room image, restricted to wherever `mask` says is real floor.

        Raises:
            CompositingError: malformed/mismatched input arrays.
        """
        room_image = self._validate_room_image(preprocessed)
        floor_mask = self._validate_mask(mask, room_image.shape[:2])
        material_rgb, material_alpha = self._validate_lighting_result(lighting_result)

        room_bounds, canvas_bounds = self._compute_overlap(
            room_image.shape[:2], lighting_result.canvas_offset, material_rgb.shape[:2]
        )

        composited = room_image.copy()
        applied_pixel_count = 0

        if room_bounds is not None:
            room_y0, room_y1, room_x0, room_x1 = room_bounds
            canvas_y0, canvas_y1, canvas_x0, canvas_x1 = canvas_bounds

            room_crop = room_image[room_y0:room_y1, room_x0:room_x1]
            mask_crop = floor_mask[room_y0:room_y1, room_x0:room_x1]
            material_crop = material_rgb[canvas_y0:canvas_y1, canvas_x0:canvas_x1]
            alpha_crop = material_alpha[canvas_y0:canvas_y1, canvas_x0:canvas_x1]

            # Intersection of the material's own alpha and the true
            # floor mask — see module docstring. np.minimum is the
            # correct intersection operator for two [0, 255]
            # opacity-like channels (and reduces to a logical AND when
            # both happen to be strictly binary, which the floor mask
            # always is).
            combined_alpha = np.minimum(alpha_crop, mask_crop)

            blended_crop = self._blend(room_crop, material_crop, combined_alpha)
            composited[room_y0:room_y1, room_x0:room_x1] = blended_crop

            applied_pixel_count = int(np.count_nonzero(combined_alpha))

        return CompositingResult(
            composited_image=composited,
            output_size=preprocessed.original_size,
            applied_pixel_count=applied_pixel_count,
        )

    # -- validation --------------------------------------------------------

    def _validate_room_image(self, preprocessed: PreprocessedImage) -> np.ndarray:
        room_image = preprocessed.original_image

        if not isinstance(room_image, np.ndarray) or room_image.ndim != 3 or room_image.shape[2] != 3:
            raise CompositingError(
                "compositing",
                f"PreprocessedImage.original_image must be an HxWx3 array; got "
                f"{getattr(room_image, 'shape', type(room_image))}.",
            )

        if room_image.dtype != np.uint8:
            raise CompositingError(
                "compositing",
                f"PreprocessedImage.original_image must be uint8; got dtype {room_image.dtype}.",
            )

        if room_image.shape[0] == 0 or room_image.shape[1] == 0:
            raise CompositingError(
                "compositing",
                f"PreprocessedImage.original_image has an empty dimension: shape {room_image.shape}.",
            )

        width, height = preprocessed.original_size
        if room_image.shape[:2] != (height, width):
            raise CompositingError(
                "compositing",
                f"PreprocessedImage.original_image shape {room_image.shape[:2]} does not "
                f"match original_size {(width, height)} (expected (height, width)).",
            )

        return room_image

    def _validate_mask(self, mask: ProcessedMask, room_shape: tuple[int, int]) -> np.ndarray:
        floor_mask = mask.binary_mask

        if not isinstance(floor_mask, np.ndarray) or floor_mask.ndim != 2:
            raise CompositingError(
                "compositing",
                f"ProcessedMask.binary_mask must be an HxW array; got "
                f"{getattr(floor_mask, 'shape', type(floor_mask))}.",
            )

        if floor_mask.dtype != np.uint8:
            raise CompositingError(
                "compositing",
                f"ProcessedMask.binary_mask must be uint8; got dtype {floor_mask.dtype}.",
            )

        if floor_mask.shape != room_shape:
            raise CompositingError(
                "compositing",
                f"ProcessedMask.binary_mask shape {floor_mask.shape} does not match the "
                f"room image's spatial dimensions {room_shape}.",
            )

        return floor_mask

    def _validate_lighting_result(
        self, lighting_result: LightingResult
    ) -> tuple[np.ndarray, np.ndarray]:
        rgb = lighting_result.adjusted_rgb
        alpha = lighting_result.alpha

        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise CompositingError(
                "compositing",
                f"LightingResult.adjusted_rgb must be an HxWx3 array; got "
                f"{getattr(rgb, 'shape', type(rgb))}.",
            )

        if rgb.dtype != np.uint8:
            raise CompositingError(
                "compositing", f"LightingResult.adjusted_rgb must be uint8; got dtype {rgb.dtype}."
            )

        if rgb.shape[0] == 0 or rgb.shape[1] == 0:
            raise CompositingError(
                "compositing",
                f"LightingResult.adjusted_rgb has an empty dimension: shape {rgb.shape}.",
            )

        if not isinstance(alpha, np.ndarray) or alpha.ndim != 2:
            raise CompositingError(
                "compositing",
                f"LightingResult.alpha must be an HxW array; got "
                f"{getattr(alpha, 'shape', type(alpha))}.",
            )

        if alpha.dtype != np.uint8:
            raise CompositingError(
                "compositing", f"LightingResult.alpha must be uint8; got dtype {alpha.dtype}."
            )

        if alpha.shape != rgb.shape[:2]:
            raise CompositingError(
                "compositing",
                f"LightingResult.alpha shape {alpha.shape} does not match adjusted_rgb's "
                f"spatial dimensions {rgb.shape[:2]}.",
            )

        if lighting_result.canvas_size != (rgb.shape[1], rgb.shape[0]):
            raise CompositingError(
                "compositing",
                f"LightingResult.canvas_size {lighting_result.canvas_size} does not match "
                f"adjusted_rgb's actual shape {rgb.shape[:2]} (width, height).",
            )

        return rgb, alpha

    # -- overlap / blending --------------------------------------------------

    def _compute_overlap(
        self,
        room_shape: tuple[int, int],
        canvas_offset: tuple[int, int],
        canvas_shape: tuple[int, int],
    ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        """Intersect the canvas's placement rectangle with the room
        image's own bounds. Returns (None, None) if there is no
        overlap at all — never raises for this, since an
        out-of-frame/degenerate placement is handled gracefully (see
        module docstring)."""
        room_height, room_width = room_shape
        canvas_height, canvas_width = canvas_shape
        offset_x, offset_y = canvas_offset

        room_x0 = max(0, offset_x)
        room_y0 = max(0, offset_y)
        room_x1 = min(room_width, offset_x + canvas_width)
        room_y1 = min(room_height, offset_y + canvas_height)

        if room_x1 <= room_x0 or room_y1 <= room_y0:
            return None, None

        canvas_x0 = room_x0 - offset_x
        canvas_y0 = room_y0 - offset_y
        canvas_x1 = room_x1 - offset_x
        canvas_y1 = room_y1 - offset_y

        return (room_y0, room_y1, room_x0, room_x1), (canvas_y0, canvas_y1, canvas_x0, canvas_x1)

    def _blend(
        self, room_crop: np.ndarray, material_crop: np.ndarray, combined_alpha: np.ndarray
    ) -> np.ndarray:
        alpha_norm = (combined_alpha.astype(np.float64) / 255.0)[..., np.newaxis]
        room_f = room_crop.astype(np.float64)
        material_f = material_crop.astype(np.float64)

        blended = room_f * (1.0 - alpha_norm) + material_f * alpha_norm
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)
