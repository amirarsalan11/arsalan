"""
Material Processor: validates and normalizes a material/texture image
into a predictable internal representation the future Perspective
stage will consume.

Named `MaterialProcessor` rather than "MaterialApplier"/"Applicator"
deliberately: this stage does NOT apply the material to anything.
Per the milestone scope, it only validates, decodes, and normalizes
the raw texture — no perspective warp, no floor-plane fitting, no
compositing. Naming it "Processor" (mirroring MaskProcessor's own
naming rationale: "processes the input into a normalized form")
avoids overclaiming a capability this class doesn't have.

Deliberate non-coupling: this stage's `process()` takes ONLY the raw
material bytes. It does not take (and does not know about)
FloorGeometry or ProcessedMask. The milestone's stated inputs list
mentions the room image / floor mask / geometry alongside the material
texture, but this stage's actual responsibility is narrowly the
texture itself — pairing a PreparedMaterial with a FloorGeometry to
actually perform a perspective warp is precisely the future
Perspective stage's job, not this one's. Coupling them here would
force this stage to anticipate Perspective's not-yet-designed API,
which is the premature coupling this milestone is explicitly scoped
to avoid.

Intended future integration point: a future Perspective stage would
take a `PreparedMaterial` (this module's output) together with a
`FloorGeometry` (geometry.py's output) and produce the actual warped
texture. `RenderPipeline.run()` is deliberately NOT modified in this
milestone to call this stage — its current signature only accepts
`image_bytes` for the room photo, with no parameter for a material
texture at all. Wiring Material (and eventually Perspective) into
`RenderPipeline` is left for whichever milestone actually implements
Perspective, once the shape of that integration is concretely known.
Until then, `MaterialProcessor` is used standalone.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from ai_engine.exceptions import ImageTooLargeError, InvalidImageError
from ai_engine.preprocessing.preprocessor import MAX_SOURCE_DIMENSION, SUPPORTED_FORMATS
from ai_engine.schemas import PreparedMaterial

# Material textures are frequently small, tileable swatches (unlike
# room photos), so the minimum accepted dimension is intentionally
# much smaller than the Preprocessor's MIN_SOURCE_DIMENSION.
MIN_TEXTURE_DIMENSION = 8

# Normalization target: textures larger than this on either side are
# downscaled (never upscaled) to this cap, preserving aspect ratio.
# This is a size-management convenience for later stages (tiling,
# perspective warping), not a hard validation limit — MAX_SOURCE_DIMENSION
# (imported from preprocessor.py) remains the actual rejection ceiling.
MAX_NORMALIZED_DIMENSION = 1024

# PIL modes that carry a genuine alpha/transparency channel. "P" (palette)
# is included conditionally — see _has_alpha_channel — since a palette
# image only has real transparency if its info dict declares it.
_DIRECT_ALPHA_MODES = frozenset({"RGBA", "LA", "PA"})


class MaterialProcessor:
    """Stateless — safe to reuse a single instance across requests."""

    def process(self, material_bytes: bytes) -> PreparedMaterial:
        """Validate and normalize a raw material/texture image.

        Raises:
            InvalidImageError: unsupported/corrupt format, or
                dimensions below MIN_TEXTURE_DIMENSION.
            ImageTooLargeError: dimensions exceed MAX_SOURCE_DIMENSION.
        """
        pil_image, source_format = self._decode(material_bytes)

        rgb, alpha, has_alpha = self._extract_channels(pil_image)
        width, height = rgb.shape[1], rgb.shape[0]
        self._validate_dimensions(width, height)

        normalized_rgb, normalized_alpha, normalized_size, was_downscaled = self._normalize(
            rgb, alpha, (width, height)
        )

        return PreparedMaterial(
            texture_rgb=normalized_rgb,
            alpha_mask=normalized_alpha,
            has_alpha=has_alpha,
            original_size=(width, height),
            normalized_size=normalized_size,
            source_format=source_format,
            was_downscaled=was_downscaled,
        )

    # -- internal steps --------------------------------------------------

    def _decode(self, material_bytes: bytes) -> tuple[Image.Image, str]:
        if not material_bytes:
            raise InvalidImageError("material", "Received empty material image data.")

        try:
            pil_image = Image.open(io.BytesIO(material_bytes))
            pil_image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError(
                "material",
                "Could not decode material image data — unsupported or corrupt file.",
                exc,
            ) from exc

        source_format = (pil_image.format or "").upper()
        if source_format not in SUPPORTED_FORMATS:
            raise InvalidImageError(
                "material",
                f"Unsupported material image format '{source_format or 'unknown'}'. "
                f"Supported formats: {sorted(SUPPORTED_FORMATS)}.",
            )

        return pil_image, source_format

    def _has_alpha_channel(self, pil_image: Image.Image) -> bool:
        if pil_image.mode in _DIRECT_ALPHA_MODES:
            return True
        # A palette ("P" mode) image only carries real transparency if
        # its info dict declares a transparency index/mask — plain
        # palette images without that key are fully opaque.
        if pil_image.mode == "P" and "transparency" in pil_image.info:
            return True
        return False

    def _extract_channels(self, pil_image: Image.Image) -> tuple[np.ndarray, np.ndarray, bool]:
        """Split the decoded image into an RGB array and a separate
        alpha array, never discarding genuine transparency.

        If the source has no alpha channel at all, a synthesized
        fully-opaque (255) alpha array is returned — this fills in the
        naturally-implied value for an opaque image rather than
        discarding any real information, and keeps `PreparedMaterial`'s
        `alpha_mask` field uniformly present for downstream consumers.
        `has_alpha` distinguishes "genuinely transparent" from
        "synthesized opaque" for anything that cares about the
        distinction.
        """
        has_alpha = self._has_alpha_channel(pil_image)

        if has_alpha:
            rgba = np.array(pil_image.convert("RGBA"), dtype=np.uint8)
            rgb = rgba[..., :3]
            alpha = rgba[..., 3]
        else:
            rgb = np.array(pil_image.convert("RGB"), dtype=np.uint8)
            alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)

        return rgb, alpha, has_alpha

    def _validate_dimensions(self, width: int, height: int) -> None:
        if width < MIN_TEXTURE_DIMENSION or height < MIN_TEXTURE_DIMENSION:
            raise InvalidImageError(
                "material",
                f"Material image dimensions {width}x{height} are below the "
                f"minimum of {MIN_TEXTURE_DIMENSION}px on each side.",
            )

        if width > MAX_SOURCE_DIMENSION or height > MAX_SOURCE_DIMENSION:
            raise ImageTooLargeError(
                "material",
                f"Material image dimensions {width}x{height} exceed the "
                f"maximum of {MAX_SOURCE_DIMENSION}px on each side.",
            )

    def _normalize(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray,
        original_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int], bool]:
        """Downscale-only, aspect-ratio-preserving normalization.

        The scale factor is capped at 1.0 (`min(1.0, ...)`) so this
        can never upscale — avoiding both uncontrolled upscaling and
        the quality loss/blur that would introduce. A texture already
        smaller than MAX_NORMALIZED_DIMENSION is returned unchanged.
        """
        width, height = original_size
        scale = min(1.0, MAX_NORMALIZED_DIMENSION / max(width, height))

        if scale >= 1.0:
            return rgb, alpha, (width, height), False

        normalized_width = max(1, round(width * scale))
        normalized_height = max(1, round(height * scale))

        # INTER_AREA is the recommended OpenCV interpolation for
        # downscaling — it averages source pixels into each destination
        # pixel, which is both deterministic and avoids the aliasing a
        # naive point-sample resize would introduce.
        normalized_rgb = cv2.resize(
            rgb, (normalized_width, normalized_height), interpolation=cv2.INTER_AREA
        )
        normalized_alpha = cv2.resize(
            alpha, (normalized_width, normalized_height), interpolation=cv2.INTER_AREA
        )

        return normalized_rgb, normalized_alpha, (normalized_width, normalized_height), True
