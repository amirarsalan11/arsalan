"""
Preprocessor: image format validation, EXIF orientation correction,
original-resolution preservation, and letterboxed 512x512 model-input
generation.

See ai_engine/schemas.py:PreprocessedImage for the output shape, and
the architecture plan for why letterboxing (not squash-resize) is used
for the model input.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ai_engine.exceptions import ImageTooLargeError, InvalidImageError
from ai_engine.schemas import LetterboxMetadata, PreprocessedImage

MODEL_INPUT_SIZE = 512
SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})

# Defensive guard against pathological memory usage. This service is
# separately deployable and cannot assume the backend's own upload
# size limits were already enforced upstream.
MAX_SOURCE_DIMENSION = 8000
MIN_SOURCE_DIMENSION = 32

# Padding fill value for the letterboxed model input (mid-gray is a
# neutral choice that doesn't bias the segmentation model toward
# either a "floor" or "not-floor" prediction in the padded region).
_LETTERBOX_FILL_VALUE = 128


class Preprocessor:
    """Stateless — safe to reuse a single instance across requests."""

    def process(self, image_bytes: bytes) -> PreprocessedImage:
        """Validate and preprocess raw image bytes.

        Raises:
            InvalidImageError: unsupported/corrupt format.
            ImageTooLargeError: dimensions exceed MAX_SOURCE_DIMENSION,
                or fall below MIN_SOURCE_DIMENSION.
        """
        pil_image, source_format = self._decode(image_bytes)
        pil_image = self._correct_orientation(pil_image)
        pil_image = pil_image.convert("RGB")

        width, height = pil_image.size
        self._validate_dimensions(width, height)

        original_image = np.array(pil_image, dtype=np.uint8)
        model_input, letterbox = self._letterbox_resize(original_image)

        return PreprocessedImage(
            original_image=original_image,
            original_size=(width, height),
            model_input=model_input,
            letterbox=letterbox,
            source_format=source_format,
        )

    # -- internal steps --------------------------------------------------

    def _decode(self, image_bytes: bytes) -> tuple[Image.Image, str]:
        if not image_bytes:
            raise InvalidImageError("preprocessor", "Received empty image data.")

        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            pil_image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError(
                "preprocessor", "Could not decode image data — unsupported or corrupt file.", exc
            ) from exc

        source_format = (pil_image.format or "").upper()
        if source_format not in SUPPORTED_FORMATS:
            raise InvalidImageError(
                "preprocessor",
                f"Unsupported image format '{source_format or 'unknown'}'. "
                f"Supported formats: {sorted(SUPPORTED_FORMATS)}.",
            )

        return pil_image, source_format

    def _correct_orientation(self, pil_image: Image.Image) -> Image.Image:
        # exif_transpose applies the EXIF Orientation tag physically to
        # the pixel data and returns an image with that tag cleared, so
        # it can never be double-applied by a later consumer.
        return ImageOps.exif_transpose(pil_image)

    def _validate_dimensions(self, width: int, height: int) -> None:
        if width < MIN_SOURCE_DIMENSION or height < MIN_SOURCE_DIMENSION:
            raise InvalidImageError(
                "preprocessor",
                f"Image dimensions {width}x{height} are below the minimum "
                f"of {MIN_SOURCE_DIMENSION}px on each side.",
            )

        if width > MAX_SOURCE_DIMENSION or height > MAX_SOURCE_DIMENSION:
            raise ImageTooLargeError(
                "preprocessor",
                f"Image dimensions {width}x{height} exceed the maximum "
                f"of {MAX_SOURCE_DIMENSION}px on each side.",
            )

    def _letterbox_resize(self, image: np.ndarray) -> tuple[np.ndarray, LetterboxMetadata]:
        """Resize preserving aspect ratio, then pad to a fixed square.

        Chosen over a naive squash-resize specifically because
        distorting the aspect ratio would corrupt the straight-line
        geometry (Hough lines / vanishing point) the Geometry Estimator
        computes later — that math assumes real-world straight lines
        stay straight, which a non-uniform squash would violate.
        """
        height, width = image.shape[:2]
        scale = min(MODEL_INPUT_SIZE / width, MODEL_INPUT_SIZE / height)

        scaled_width = max(1, round(width * scale))
        scaled_height = max(1, round(height * scale))

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (scaled_width, scaled_height), interpolation=interpolation)

        pad_total_x = MODEL_INPUT_SIZE - scaled_width
        pad_total_y = MODEL_INPUT_SIZE - scaled_height
        pad_left = pad_total_x // 2
        pad_right = pad_total_x - pad_left
        pad_top = pad_total_y // 2
        pad_bottom = pad_total_y - pad_top

        letterboxed = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(_LETTERBOX_FILL_VALUE,) * 3,
        )

        metadata = LetterboxMetadata(
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=pad_right,
            pad_bottom=pad_bottom,
        )
        return letterboxed, metadata
