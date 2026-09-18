"""
Mask Processor: upscales the model's 512x512 floor probability mask
back to original resolution, cleans it up (morphological opening/
closing, hole filling, largest-component selection), and applies
bilateral edge smoothing.

See ai_engine/schemas.py:ProcessedMask for the output shape.
"""

from __future__ import annotations

import cv2
import numpy as np

from ai_engine.schemas import LetterboxMetadata, ProcessedMask

# Probability threshold for converting the continuous [0,1] mask into
# a binary decision, applied both right after inverse-letterboxing and
# again after smoothing (smoothing can blur values toward the
# boundary, so a final re-threshold keeps the output strictly binary).
_CONFIDENCE_THRESHOLD = 0.5

# Morphological kernel size is expressed as a fraction of the image
# diagonal so behavior generalizes across wildly different input
# resolutions rather than assuming a fixed pixel count.
_MORPH_KERNEL_DIAGONAL_FRACTION = 0.01
_MIN_MORPH_KERNEL_SIZE = 3

# Bilateral filter parameters. Diameter is resolution-relative (like
# the morphological kernel); sigmaColor/sigmaSpace are fixed since
# they operate on the 0-255 intensity scale of the mask itself, not on
# absolute pixel distances that would need to scale with resolution.
_BILATERAL_DIAMETER_DIAGONAL_FRACTION = 0.01
_MIN_BILATERAL_DIAMETER = 5
_BILATERAL_SIGMA_COLOR = 50.0
_BILATERAL_SIGMA_SPACE = 50.0


class MaskProcessor:
    """Stateless — safe to reuse a single instance across requests."""

    def process(
        self,
        raw_mask: np.ndarray,
        original_size: tuple[int, int],
        letterbox: LetterboxMetadata,
    ) -> ProcessedMask:
        """
        Args:
            raw_mask: 512x512 float32 array, per-pixel floor probability [0,1].
            original_size: (width, height) of the source image.
            letterbox: metadata recorded by the Preprocessor, used to
                invert the letterbox transform before upscaling.
        """
        upscaled = self._upscale_to_original(raw_mask, original_size, letterbox)
        cleaned, discarded_count = self._morphological_cleanup(upscaled)
        smoothed = self._bilateral_smooth(cleaned)

        area_ratio = float(np.count_nonzero(smoothed)) / (original_size[0] * original_size[1])
        bounding_box = self._bounding_box(smoothed)

        return ProcessedMask(
            binary_mask=smoothed,
            area_ratio=area_ratio,
            bounding_box=bounding_box,
            discarded_region_count=discarded_count,
        )

    # -- internal steps --------------------------------------------------

    def _upscale_to_original(
        self,
        raw_mask: np.ndarray,
        original_size: tuple[int, int],
        letterbox: LetterboxMetadata,
    ) -> np.ndarray:
        """Invert the letterbox transform, then resize the remaining
        central region up to the original resolution."""
        width, height = original_size
        model_size = raw_mask.shape[0]

        # Crop out the padding that was added by the Preprocessor.
        top = letterbox.pad_top
        bottom = model_size - letterbox.pad_bottom
        left = letterbox.pad_left
        right = model_size - letterbox.pad_right
        cropped = raw_mask[top:bottom, left:right]

        # INTER_LINEAR (not NEAREST) for a smoother upscale, then
        # re-threshold immediately after — this avoids the blocky,
        # staircase edges a nearest-neighbor upscale of a categorical
        # mask would otherwise produce.
        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        binary = np.where(resized >= _CONFIDENCE_THRESHOLD, 255, 0).astype(np.uint8)
        return binary

    def _morphological_cleanup(self, binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
        height, width = binary_mask.shape[:2]
        diagonal = float(np.hypot(width, height))

        kernel_size = max(_MIN_MORPH_KERNEL_SIZE, round(diagonal * _MORPH_KERNEL_DIAGONAL_FRACTION))
        if kernel_size % 2 == 0:
            kernel_size += 1  # odd kernel sizes are well-defined/centered for morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        filled = self._fill_holes(closed)

        return self._keep_largest_component(filled)

    def _fill_holes(self, binary_mask: np.ndarray) -> np.ndarray:
        """Flood-fill from the border to find background-connected
        zero-regions; anything the flood-fill can't reach is an
        enclosed hole and gets filled to foreground."""
        height, width = binary_mask.shape[:2]

        # floodFill needs a mask 2px larger than the image on each side.
        flood_fill_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        inverted = cv2.bitwise_not(binary_mask)
        flooded = inverted.copy()

        # connectivity=8 avoids leaving diagonal-only-connected
        # background pockets mistaken for enclosed holes.
        cv2.floodFill(flooded, flood_fill_mask, (0, 0), 0, flags=8)

        # After this call, `flooded` is 255 ONLY at pixels that were
        # background in `binary_mask` but NOT reachable from the image
        # border by flood fill — i.e. exactly the enclosed holes.
        # Everywhere else (original floor pixels, and border-connected
        # background) is 0. OR-ing it back into `binary_mask` fills
        # precisely those enclosed holes and nothing else.
        filled = cv2.bitwise_or(binary_mask, flooded)
        return filled

    def _keep_largest_component(self, binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
        num_labels, labels = cv2.connectedComponents(binary_mask)

        if num_labels <= 1:
            # Label 0 is background; no foreground components at all.
            return np.zeros_like(binary_mask), 0

        # Component sizes, excluding background label 0.
        sizes = [int(np.count_nonzero(labels == label)) for label in range(1, num_labels)]
        largest_label = 1 + int(np.argmax(sizes))
        discarded_count = (num_labels - 1) - 1  # total foreground components minus the one kept

        largest_only = np.where(labels == largest_label, 255, 0).astype(np.uint8)
        return largest_only, discarded_count

    def _bilateral_smooth(self, binary_mask: np.ndarray) -> np.ndarray:
        height, width = binary_mask.shape[:2]
        diagonal = float(np.hypot(width, height))

        diameter = max(
            _MIN_BILATERAL_DIAMETER, round(diagonal * _BILATERAL_DIAMETER_DIAGONAL_FRACTION)
        )
        if diameter % 2 == 0:
            diameter += 1

        smoothed = cv2.bilateralFilter(
            binary_mask, diameter, _BILATERAL_SIGMA_COLOR, _BILATERAL_SIGMA_SPACE
        )
        # Bilateral filtering operates on intensity, so re-threshold
        # afterward to keep the output strictly binary — smoothing is
        # meant to reduce jagged aliasing at the boundary, not
        # introduce genuinely continuous alpha values (that's the
        # future Compositor's job, with its own feathering pass).
        rethresholded = np.where(smoothed >= 128, 255, 0).astype(np.uint8)
        return rethresholded

    def _bounding_box(self, binary_mask: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(binary_mask)
        if xs.size == 0 or ys.size == 0:
            return (0, 0, 0, 0)

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
