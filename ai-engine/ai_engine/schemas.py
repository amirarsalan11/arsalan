"""
Shared data schemas for the AI pipeline.

Single source of truth for every dataclass passed between pipeline
stages (Preprocessor -> Segmentor -> Mask Processor -> Geometry
Estimator -> Pipeline Orchestrator).

Design choice: plain `dataclass(frozen=True, slots=True)`, not
Pydantic. These objects carry large NumPy arrays (full-resolution
images, masks) passed stage-to-stage on every request; Pydantic's
runtime validation overhead and awkward handling of arbitrary NumPy
types isn't a good fit for this internal, performance-sensitive,
process-local boundary. Pydantic remains the right tool at the FastAPI
boundary (see backend/app/schemas/) — this is a different boundary.

`frozen=True` enforces that no stage ever mutates another stage's
output in place; `slots=True` avoids per-instance __dict__ overhead
given how many of these objects get created per request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# --- shared metadata -------------------------------------------------


@dataclass(frozen=True, slots=True)
class LetterboxMetadata:
    """Records exactly how the original image was resized+padded to
    the fixed 512x512 model input, so that transformation can be
    inverted precisely when upscaling the mask back to original
    resolution."""

    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


# --- stage 1: Preprocessor output ------------------------------------


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    """Output of preprocessor.py."""

    original_image: np.ndarray  # HxWx3 uint8 RGB, full original resolution
    original_size: tuple[int, int]  # (width, height)
    model_input: np.ndarray  # 512x512x3 uint8 RGB, letterboxed, NOT normalized
    letterbox: LetterboxMetadata
    source_format: str  # "JPEG" | "PNG" | "WEBP"


# --- stage 2: Segmentor output ----------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """Output of segmentor.py."""

    raw_mask: np.ndarray  # 512x512 float32, per-pixel floor probability [0,1]
    mean_confidence: float  # aggregate confidence within the predicted floor region
    model_version: str  # checkpoint identifier, for future debugging/A-B comparisons


# --- stage 3: Mask Processor output ------------------------------------


@dataclass(frozen=True, slots=True)
class ProcessedMask:
    """Output of mask_processor.py."""

    binary_mask: np.ndarray  # original_size resolution, uint8 {0, 255}
    area_ratio: float  # mask area / (width * height)
    bounding_box: tuple[int, int, int, int]  # x, y, w, h of the retained largest component
    discarded_region_count: int  # how many smaller components were dropped


# --- stage 4: Geometry Estimator output ---------------------------------


@dataclass(frozen=True, slots=True)
class LineSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    angle_deg: float


@dataclass(frozen=True, slots=True)
class FloorGeometry:
    """Output of geometry.py."""

    contour: np.ndarray  # Nx2 points, full contour
    simplified_polygon: np.ndarray  # Mx2 points, M << N
    corners: tuple[tuple[int, int], ...]  # 4 ordered points (TL, TR, BR, BL)
    lines: list[LineSegment]  # classified structural lines used
    vanishing_point: tuple[float, float] | None
    geometry_confidence: float  # 0.0-1.0; lower if vanishing point absent, etc.


# --- stage 5: Material Processor output ---------------------------------


@dataclass(frozen=True, slots=True)
class PreparedMaterial:
    """Output of material/material_processor.py.

    A validated, normalized representation of a material texture,
    ready for a future Perspective stage to warp onto the floor plane
    described by FloorGeometry. This stage does NOT perform that warp,
    and does NOT itself consume FloorGeometry/ProcessedMask — see
    material_processor.py's module docstring for why that coupling is
    deliberately deferred to whichever future stage actually needs
    both a geometry and a material at once.
    """

    texture_rgb: np.ndarray  # HxWx3 uint8 RGB, normalized texture (alpha not baked in)
    alpha_mask: np.ndarray  # HxW uint8, 255=fully opaque; synthesized as all-255 if the source had no alpha channel
    has_alpha: bool  # True only if the source image itself carried a genuine alpha channel
    original_size: tuple[int, int]  # (width, height) of the source texture, before normalization
    normalized_size: tuple[int, int]  # (width, height) after normalization
    source_format: str  # "JPEG" | "PNG" | "WEBP"
    was_downscaled: bool  # True if normalization reduced the texture's size; normalization never upscales


# --- stage 6: Perspective Transformer output -----------------------------


@dataclass(frozen=True, slots=True)
class PerspectiveTransformResult:
    """Output of perspective/perspective_transformer.py.

    A material texture warped onto the floor-plane quadrilateral
    described by FloorGeometry.corners — expressed in the canvas's OWN
    local coordinate space, not the full room image's. See
    perspective_transformer.py's module docstring for why: this stage
    only receives PreparedMaterial + FloorGeometry (neither of which
    carries the room image's own width/height), so it warps into a
    tight bounding box around just the floor corners (plus a small
    margin) rather than a full-room-sized canvas. `canvas_offset`
    records where that box sits within the original room image's
    coordinate space, so a future Compositing stage can paste it back
    at the right position.
    """

    warped_rgb: np.ndarray  # canvas_height x canvas_width x 3 uint8 RGB
    warped_alpha: np.ndarray  # canvas_height x canvas_width uint8; 0 outside the floor quad, follows material's alpha inside it
    homography_matrix: np.ndarray  # 3x3 float64: material-rectangle -> local floor quad
    canvas_size: tuple[int, int]  # (width, height) of warped_rgb / warped_alpha
    canvas_offset: tuple[int, int]  # (x, y) of this canvas's top-left within the original room image
    destination_corners: tuple[tuple[int, int], ...]  # the 4 floor corners, shifted into this canvas's local space (TL, TR, BR, BL)
    source_size: tuple[int, int]  # (width, height) of the material rectangle used as the homography source
    source_has_alpha: bool  # passthrough of PreparedMaterial.has_alpha, for traceability


# --- stage 7: Lighting Processor output ----------------------------------


@dataclass(frozen=True, slots=True)
class LightingResult:
    """Output of lighting/lighting_processor.py.

    A deterministic exposure/gain adjustment of a
    PerspectiveTransformResult's warped material, toward either a
    fixed neutral mid-gray target or an explicit caller-supplied
    target mean luminance. See lighting_processor.py's module
    docstring for why no other "room lighting" signal is used: the
    current architecture provides no explicit lighting map, and this
    stage does not invent one.

    `alpha` and `canvas_size`/`canvas_offset` are carried over
    unchanged from the input PerspectiveTransformResult, so a future
    Compositing stage can consume LightingResult on its own without
    also needing to hold onto the earlier PerspectiveTransformResult
    just to recover this same, already-established metadata.
    """

    adjusted_rgb: np.ndarray  # same shape/dtype as the input warped_rgb, uint8
    alpha: np.ndarray  # unchanged passthrough of the input warped_alpha
    canvas_size: tuple[int, int]  # passthrough of PerspectiveTransformResult.canvas_size
    canvas_offset: tuple[int, int]  # passthrough of PerspectiveTransformResult.canvas_offset
    applied_gain: float  # the multiplicative RGB scale factor actually applied (1.0 = no change)
    target_mean_luminance: float  # the target actually used (caller-supplied, or the MVP default)


# --- orchestration-level records ----------------------------------------


@dataclass(frozen=True, slots=True)
class StageTiming:
    stage_name: str
    duration_ms: float


@dataclass(frozen=True, slots=True)
class PipelineWarning:
    stage: str
    code: str  # e.g. "MULTIPLE_REGIONS_FOUND", "VANISHING_POINT_UNRESOLVED"
    message: str


PipelineStatus = Literal["ok", "no_floor_detected", "low_confidence", "geometry_degraded"]


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Final result of pipeline.py's RenderPipeline.run()."""

    status: PipelineStatus
    preprocessed: PreprocessedImage
    segmentation: SegmentationResult
    mask: ProcessedMask
    geometry: FloorGeometry | None  # None if short-circuited before this stage ran
    timings: list[StageTiming] = field(default_factory=list)
    warnings: list[PipelineWarning] = field(default_factory=list)
