"""
Exception hierarchy for the AI pipeline.

Every stage raises a subclass of `PipelineStageError` rather than a
bare/builtin exception, so the Pipeline Orchestrator (and, eventually,
a Celery task calling into this package) has exactly one exception
type to catch at the top level, while still being able to inspect
`stage` and `original_exception` for diagnosis, and to distinguish
semantically different failure modes (e.g. "this photo has no floor"
vs. "the model failed to load") that warrant different handling.
"""

from __future__ import annotations


class PipelineStageError(Exception):
    """Base class for all AI-pipeline errors.

    Carries the name of the stage that raised it and, where
    applicable, the original underlying exception that was caught and
    wrapped.
    """

    def __init__(self, stage: str, message: str, original_exception: Exception | None = None) -> None:
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message
        self.original_exception = original_exception


class InvalidImageError(PipelineStageError):
    """Raised by the Preprocessor when the input cannot be decoded as
    a supported image format, or fails basic sanity checks."""


class ImageTooLargeError(PipelineStageError):
    """Raised by the Preprocessor when the source image's dimensions
    exceed a configured maximum, as a defensive guard against
    pathological memory usage. The AI engine is a separately
    deployable service and cannot assume an upstream caller already
    enforced size limits."""


class ModelLoadError(PipelineStageError):
    """Raised by the model registry when segmentation model weights
    fail to load. Fails fast and loudly at first use, rather than
    surfacing as a mysterious failure deep inside a request."""


class InferenceError(PipelineStageError):
    """Raised by the Segmentor when the forward pass itself fails
    (e.g. out-of-memory). Deliberately distinct from
    NoFloorDetectedError: this is a transient/resource failure a
    caller might reasonably retry, whereas "no floor in this photo"
    never benefits from a retry."""


class NoFloorDetectedError(PipelineStageError):
    """Raised (or, more commonly, surfaced via PipelineResult.status
    rather than raised — see pipeline.py) when segmentation/mask
    processing finds no plausible floor region in the image."""


class GeometryEstimationError(PipelineStageError):
    """Raised by the Geometry Estimator when even a degraded,
    polygon-only estimate cannot be produced (e.g. the mask's contour
    is degenerate/empty). Graceful degradation is preferred wherever
    possible; this is reserved for the case where there is truly
    nothing usable to return."""


class PerspectiveTransformError(PipelineStageError):
    """Raised by the Perspective stage when FloorGeometry's corners
    are missing, malformed, degenerate (duplicate/collinear points,
    zero-area quadrilateral), or when the resulting homography is
    numerically unusable (non-finite values or a near-singular
    matrix). Also raised for malformed PreparedMaterial dimensions.

    Distinct from GeometryEstimationError: this stage receives an
    ALREADY-PRODUCED FloorGeometry and validates it defensively (it
    must not blindly trust it), rather than estimating geometry
    itself — the failure mode here is "this geometry can't be turned
    into a usable transform," not "no geometry could be estimated."
    """
