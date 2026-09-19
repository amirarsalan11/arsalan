"""
Lighting Processor: a deterministic, MVP exposure/gain normalization
of a perspective-warped material, so its overall brightness can be
brought toward a target before a future Compositing stage blends it
into the room photo.

===========================================================================
ARCHITECTURE CHECK — what lighting information is actually available
===========================================================================
Before writing this stage, the existing schemas were inspected for any
explicit "room lighting map" or similar signal a Lighting stage could
consume. None exists:
  - `PreprocessedImage` carries the original room pixels, but nothing
    about its lighting/exposure characteristics as a derived value.
  - `SegmentationResult` / `ProcessedMask` / `FloorGeometry` are purely
    about WHERE the floor is, not how it's lit.
  - `PerspectiveTransformResult` (this stage's actual input) carries
    only the warped material itself (`warped_rgb`/`warped_alpha`) plus
    geometric metadata (`canvas_size`/`canvas_offset`/etc.) — no
    reference to the room image's own pixels or lighting statistics.

So there is no reliable, already-computed "target room lighting" value
this stage can read from the architecture as it exists today. Per
"do not invent room-lighting data the architecture doesn't provide,"
this stage does NOT reach into `PreprocessedImage`/`ProcessedMask`
itself to sample the room's floor-area lighting — that would mean
Lighting quietly depending on a much wider slice of pipeline state
than it's actually given, and deciding on its own how to sample "the
room's lighting" (a judgment call this milestone has no mandate to
make).

The chosen MVP interpretation: `process()` takes the
PerspectiveTransformResult it's actually given, plus an OPTIONAL
`target_mean_luminance: float | None`. If omitted (the default, and
the only thing exercised when this stage is used standalone), the
material's own mean luminance is normalized toward a fixed, neutral
mid-gray constant (128.0) — a deterministic, bounded, self-contained
adjustment that uses only data this stage genuinely has. If a future
caller — e.g. once Compositing exists and has both
`PreprocessedImage.original_image` and `ProcessedMask.binary_mask` in
hand from the same pipeline run — wants to match the material's
exposure to the room's actual floor-area brightness, it can compute
that mean luminance itself (from data that already, genuinely exists
elsewhere in the architecture) and pass it in as this one explicit,
optional float. This avoids adding a required, speculative coupling
to PreprocessedImage/ProcessedMask into this stage, while still making
the "match real room lighting" use case straightforwardly possible for
whichever future stage actually assembles that data.

===========================================================================
ALGORITHM
===========================================================================
1. Compute the mean luminance (standard broadcast weights: 0.299R +
   0.587G + 0.114B) over only the OPAQUE pixels (alpha > 0) of the
   warped material. Border/margin pixels outside the floor
   quadrilateral are alpha=0 and RGB=0 (per Perspective's
   borderValue=0) and must be excluded — including them would corrupt
   the statistic with meaningless black padding.
2. gain = target_mean_luminance / current_mean_luminance, clamped to
   [_MIN_GAIN, _MAX_GAIN] to prevent pathological extreme adjustments
   (e.g. a near-black material producing a huge gain that would blow
   out to pure white) — this is what "bounded so it cannot produce
   invalid pixel values" means here, on top of the final clip.
3. adjusted_rgb = clip(rgb * gain, 0, 255), computed in float64 then
   cast back to uint8 — avoids uint8 overflow wraparound during the
   multiply.
4. Alpha is passed through completely unchanged — this stage adjusts
   brightness only, never transparency/shape.

This is a plain linear exposure/gain correction — not per-channel color
grading, not a gamma/tone curve, not any attempt at physically accurate
illumination. It is deliberately the simplest deterministic operation
that satisfies "adapt material brightness toward a lighting target."

If the material has no opaque pixels at all (e.g. everything
transparent), there is nothing meaningful to normalize: the stage
applies gain=1.0 (an identity pass-through) rather than dividing by a
zero/undefined mean.
"""

from __future__ import annotations

import numpy as np

from ai_engine.exceptions import LightingAdjustmentError
from ai_engine.schemas import LightingResult, PerspectiveTransformResult

# Neutral mid-gray default target — the midpoint of the valid uint8
# range, used only when the caller doesn't supply an explicit target.
_DEFAULT_TARGET_MEAN_LUMINANCE = 128.0

# Standard broadcast/perceptual luminance weights (ITU-R BT.601), the
# same well-established formula used throughout image processing —
# not an arbitrary/invented weighting.
_LUMINANCE_WEIGHTS = (0.299, 0.587, 0.114)

# Gain is clamped to this range regardless of how extreme the
# computed ratio would otherwise be, per "bounded so it cannot produce
# invalid pixel values" and to avoid a near-zero current-luminance
# denominator producing an absurd gain.
_MIN_GAIN = 0.25
_MAX_GAIN = 4.0

# Below this mean luminance (out of 255), the material is treated as
# having no meaningful brightness signal to normalize against — gain
# is left at 1.0 rather than dividing by a near-zero denominator.
_MIN_MEANINGFUL_LUMINANCE = 1.0

_VALID_LUMINANCE_RANGE = (0.0, 255.0)


class LightingProcessor:
    """Stateless — safe to reuse a single instance across requests."""

    def process(
        self,
        perspective_result: PerspectiveTransformResult,
        target_mean_luminance: float | None = None,
    ) -> LightingResult:
        """Normalize `perspective_result`'s warped material toward a
        target mean luminance.

        Args:
            perspective_result: output of PerspectiveTransformer.transform().
            target_mean_luminance: optional explicit target in [0, 255].
                Defaults to a fixed neutral mid-gray (128.0) when omitted
                — see module docstring for why no "real room lighting"
                value is read automatically.

        Raises:
            LightingAdjustmentError: malformed input arrays, or an
                out-of-range/non-finite explicit target.
        """
        rgb, alpha = self._validate_perspective_result(perspective_result)
        target = self._validate_target(target_mean_luminance)

        gain, current_mean = self._compute_gain(rgb, alpha, target)
        adjusted_rgb = self._apply_gain(rgb, gain)

        return LightingResult(
            adjusted_rgb=adjusted_rgb,
            alpha=alpha,
            canvas_size=perspective_result.canvas_size,
            canvas_offset=perspective_result.canvas_offset,
            applied_gain=gain,
            target_mean_luminance=target,
        )

    # -- validation --------------------------------------------------------

    def _validate_perspective_result(
        self, perspective_result: PerspectiveTransformResult
    ) -> tuple[np.ndarray, np.ndarray]:
        rgb = perspective_result.warped_rgb
        alpha = perspective_result.warped_alpha

        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise LightingAdjustmentError(
                "lighting",
                f"warped_rgb must be an HxWx3 array; got "
                f"{getattr(rgb, 'shape', type(rgb))}.",
            )

        if rgb.dtype != np.uint8:
            raise LightingAdjustmentError(
                "lighting", f"warped_rgb must be uint8; got dtype {rgb.dtype}."
            )

        if rgb.shape[0] == 0 or rgb.shape[1] == 0:
            raise LightingAdjustmentError(
                "lighting", f"warped_rgb has an empty dimension: shape {rgb.shape}."
            )

        if not isinstance(alpha, np.ndarray) or alpha.ndim != 2:
            raise LightingAdjustmentError(
                "lighting",
                f"warped_alpha must be an HxW array; got "
                f"{getattr(alpha, 'shape', type(alpha))}.",
            )

        if alpha.dtype != np.uint8:
            raise LightingAdjustmentError(
                "lighting", f"warped_alpha must be uint8; got dtype {alpha.dtype}."
            )

        if alpha.shape != rgb.shape[:2]:
            raise LightingAdjustmentError(
                "lighting",
                f"warped_alpha shape {alpha.shape} does not match warped_rgb's "
                f"spatial dimensions {rgb.shape[:2]}.",
            )

        return rgb, alpha

    def _validate_target(self, target_mean_luminance: float | None) -> float:
        if target_mean_luminance is None:
            return _DEFAULT_TARGET_MEAN_LUMINANCE

        target = float(target_mean_luminance)
        low, high = _VALID_LUMINANCE_RANGE

        if not np.isfinite(target):
            raise LightingAdjustmentError(
                "lighting", f"target_mean_luminance must be finite; got {target_mean_luminance}."
            )

        if not (low <= target <= high):
            raise LightingAdjustmentError(
                "lighting",
                f"target_mean_luminance must be within [{low}, {high}]; got {target}.",
            )

        return target

    # -- gain computation / application -------------------------------------

    def _compute_gain(
        self, rgb: np.ndarray, alpha: np.ndarray, target: float
    ) -> tuple[float, float]:
        opaque = alpha > 0
        if not np.any(opaque):
            # Nothing meaningful to normalize against (e.g. a fully
            # transparent material) — identity pass-through.
            return 1.0, 0.0

        rgb_float = rgb.astype(np.float64)
        weights = np.array(_LUMINANCE_WEIGHTS, dtype=np.float64)
        luminance = rgb_float @ weights  # HxW, per-pixel luminance

        current_mean = float(luminance[opaque].mean())

        if current_mean < _MIN_MEANINGFUL_LUMINANCE:
            # Near-black material: any finite target would demand an
            # enormous gain to reach; clamping below already guards
            # this, but skip the division entirely for a cleaner,
            # explicit identity result in the degenerate case.
            return 1.0, current_mean

        gain = target / current_mean
        gain = float(np.clip(gain, _MIN_GAIN, _MAX_GAIN))
        return gain, current_mean

    def _apply_gain(self, rgb: np.ndarray, gain: float) -> np.ndarray:
        if gain == 1.0:
            # No-op path avoids an unnecessary float round-trip copy
            # when nothing would change anyway.
            return rgb.copy()

        scaled = rgb.astype(np.float64) * gain
        clipped = np.clip(scaled, 0.0, 255.0)
        return clipped.astype(np.uint8)
