"""
Pipeline Orchestrator: sequential execution of Preprocessor ->
Segmentor -> Mask Processor -> Geometry Estimator -> Material
Processor -> Perspective Transformer -> Lighting Processor ->
Compositor, with timing metrics, early-exit short-circuiting on
low-confidence/no-floor/no-geometry results, and normalized error
handling.

Pipeline Integration milestone: the four downstream stages (Material,
Perspective, Lighting, Compositing) only run when BOTH of these hold:
  1. `run()` was called with a non-None `material_bytes` — the
     pipeline's pre-existing single-argument call sites (every prior
     test, and any external caller that hasn't been updated) leave
     this at its default of `None` and get EXACTLY the previous
     behavior, unchanged.
  2. `geometry is not None` — checked on the object itself, not on
     `status`. This matters because `status == "geometry_degraded"`
     covers two different underlying cases: a real, usable
     polygon-only FloorGeometry (vanishing point just wasn't
     resolved — still perfectly fine input for Perspective), and a
     GeometryEstimationError sub-case where geometry stays None (there
     is genuinely nothing to warp onto). Gating on the object, not the
     string, is what correctly lets the first case proceed while still
     skipping downstream work for the second — and for the earlier
     no_floor_detected early exits, where geometry is always None.

This module still has zero knowledge of FastAPI, Celery, or the
database — per the architecture lock, "The AI Engine is independent
from FastAPI." It is a plain, importable Python class.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import cv2
import numpy as np

from ai_engine.compositing.compositor import Compositor
from ai_engine.exceptions import GeometryEstimationError, PipelineStageError
from ai_engine.geometry.geometry import GeometryEstimator
from ai_engine.lighting.lighting_processor import LightingProcessor
from ai_engine.mask_processing.mask_processor import MaskProcessor
from ai_engine.material.material_processor import MaterialProcessor
from ai_engine.perspective.perspective_transformer import PerspectiveTransformer
from ai_engine.preprocessing.preprocessor import Preprocessor
from ai_engine.schemas import PipelineResult, PipelineWarning, ProcessedMask, StageTiming
from ai_engine.segmentation.segmentor import Segmentor

# Below this mean confidence, the pipeline short-circuits before
# running Mask Processing / Geometry Estimation at all — proceeding
# further would just be computing structure from noise.
_MIN_SEGMENTATION_CONFIDENCE = 0.3

# Below this fraction of the frame, there isn't enough floor area to
# produce meaningful geometry (e.g. floor almost entirely occluded).
_MIN_FLOOR_AREA_RATIO = 0.03


class RenderPipeline:
    """Stateless orchestration — safe to reuse a single instance
    across requests. Each stage's own class is instantiated once and
    reused for the same reason (they are themselves stateless; the
    Segmentor's underlying model is cached separately, at the process
    level, by the model registry)."""

    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        segmentor: Segmentor | None = None,
        mask_processor: MaskProcessor | None = None,
        geometry_estimator: GeometryEstimator | None = None,
        material_processor: MaterialProcessor | None = None,
        perspective_transformer: PerspectiveTransformer | None = None,
        lighting_processor: LightingProcessor | None = None,
        compositor: Compositor | None = None,
    ) -> None:
        self._preprocessor = preprocessor if preprocessor is not None else Preprocessor()
        self._segmentor = segmentor if segmentor is not None else Segmentor()
        self._mask_processor = mask_processor if mask_processor is not None else MaskProcessor()
        self._geometry_estimator = (
            geometry_estimator if geometry_estimator is not None else GeometryEstimator()
        )
        self._material_processor = (
            material_processor if material_processor is not None else MaterialProcessor()
        )
        self._perspective_transformer = (
            perspective_transformer if perspective_transformer is not None else PerspectiveTransformer()
        )
        self._lighting_processor = (
            lighting_processor if lighting_processor is not None else LightingProcessor()
        )
        self._compositor = compositor if compositor is not None else Compositor()

    def run(self, image_bytes: bytes, material_bytes: bytes | None = None) -> PipelineResult:
        """Run the pipeline on raw room-image bytes.

        Args:
            image_bytes: the room photo to analyze.
            material_bytes: optional raw material/texture image. When
                omitted (the default), the pipeline behaves exactly as
                before this milestone — it stops after Geometry. When
                provided, and geometry turned out to be usable (see
                module docstring — checked via `geometry is not None`,
                not via `status`), the pipeline continues through
                Material -> Perspective -> Lighting -> Compositing and
                populates those four PipelineResult fields.

        Raises:
            PipelineStageError (or a subclass): if any executed stage
                fails outright. Mask Processing and Geometry Estimation
                failures are handled more gracefully where possible
                (see module docstring on short-circuiting and
                geometry.py's degradation behavior) — but a truly
                degenerate mask can still raise GeometryEstimationError,
                which is caught here and converted into a
                `geometry_degraded` status rather than propagating,
                since a photo with no floor is a normal, expected
                outcome, not a pipeline bug. Material/Perspective/
                Lighting/Compositing failures are NOT given an
                equivalent graceful-degradation path — each of those
                stages' own exceptions already represent a hard,
                unrecoverable failure by design (see their own
                docstrings), so they propagate through unchanged,
                still normalized by `_timed()` exactly like every
                earlier stage.
        """
        timings: list[StageTiming] = []
        warnings: list[PipelineWarning] = []

        with self._timed("preprocessing", timings):
            preprocessed = self._preprocessor.process(image_bytes)

        with self._timed("segmentation", timings):
            segmentation = self._segmentor.segment(preprocessed.model_input)

        if segmentation.mean_confidence < _MIN_SEGMENTATION_CONFIDENCE:
            warnings.append(
                PipelineWarning(
                    stage="segmentation",
                    code="LOW_SEGMENTATION_CONFIDENCE",
                    message=(
                        f"Mean floor confidence {segmentation.mean_confidence:.2f} is below "
                        f"the minimum of {_MIN_SEGMENTATION_CONFIDENCE}."
                    ),
                )
            )
            return PipelineResult(
                status="no_floor_detected",
                preprocessed=preprocessed,
                segmentation=segmentation,
                mask=self._empty_mask_result(preprocessed.original_size),
                geometry=None,
                timings=timings,
                warnings=warnings,
            )

        with self._timed("mask_processing", timings):
            mask = self._mask_processor.process(
                segmentation.raw_mask, preprocessed.original_size, preprocessed.letterbox
            )

        if mask.area_ratio < _MIN_FLOOR_AREA_RATIO:
            warnings.append(
                PipelineWarning(
                    stage="mask_processing",
                    code="INSUFFICIENT_FLOOR_AREA",
                    message=(
                        f"Floor area ratio {mask.area_ratio:.3f} is below the minimum of "
                        f"{_MIN_FLOOR_AREA_RATIO}."
                    ),
                )
            )
            return PipelineResult(
                status="no_floor_detected",
                preprocessed=preprocessed,
                segmentation=segmentation,
                mask=mask,
                geometry=None,
                timings=timings,
                warnings=warnings,
            )

        geometry = None
        status = "ok"
        try:
            with self._timed("geometry", timings):
                grayscale = cv2.cvtColor(preprocessed.original_image, cv2.COLOR_RGB2GRAY)
                geometry = self._geometry_estimator.estimate(mask.binary_mask, grayscale)

            if geometry.vanishing_point is None:
                warnings.append(
                    PipelineWarning(
                        stage="geometry",
                        code="VANISHING_POINT_UNRESOLVED",
                        message="Falling back to polygon-only corner estimate.",
                    )
                )
                status = "geometry_degraded"

        except GeometryEstimationError as exc:
            # A truly degenerate mask (e.g. one that survived the area
            # check but has a zero-area contour after cleanup) is a
            # normal "couldn't find usable geometry" outcome, not a
            # pipeline bug — surfaced via status, not re-raised.
            warnings.append(
                PipelineWarning(
                    stage="geometry",
                    code="GEOMETRY_ESTIMATION_FAILED",
                    message=str(exc),
                )
            )
            status = "geometry_degraded"

        if mask.discarded_region_count > 0:
            warnings.append(
                PipelineWarning(
                    stage="mask_processing",
                    code="MULTIPLE_REGIONS_FOUND",
                    message=(
                        f"{mask.discarded_region_count} smaller disconnected region(s) "
                        "were discarded in favor of the largest floor region."
                    ),
                )
            )

        material_result = None
        perspective_result = None
        lighting_result = None
        compositing_result = None

        # Gated on the geometry OBJECT, not the status string — see
        # module docstring for why a "geometry_degraded" status can
        # still carry a perfectly usable FloorGeometry.
        if geometry is not None and material_bytes is not None:
            with self._timed("material", timings):
                material_result = self._material_processor.process(material_bytes)

            with self._timed("perspective", timings):
                perspective_result = self._perspective_transformer.transform(
                    material_result, geometry
                )

            with self._timed("lighting", timings):
                lighting_result = self._lighting_processor.process(perspective_result)

            with self._timed("compositing", timings):
                compositing_result = self._compositor.composite(
                    preprocessed, mask, lighting_result
                )

        return PipelineResult(
            status=status,
            preprocessed=preprocessed,
            segmentation=segmentation,
            mask=mask,
            geometry=geometry,
            material=material_result,
            perspective=perspective_result,
            lighting=lighting_result,
            compositing=compositing_result,
            timings=timings,
            warnings=warnings,
        )

    # -- internals -----------------------------------------------------------

    @contextmanager
    def _timed(self, stage_name: str, timings: list[StageTiming]) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        except PipelineStageError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize any unexpected stage failure
            raise PipelineStageError(stage_name, "Unexpected stage failure.", exc) from exc
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            timings.append(StageTiming(stage_name=stage_name, duration_ms=duration_ms))

    def _empty_mask_result(self, original_size: tuple[int, int]) -> ProcessedMask:
        width, height = original_size
        return ProcessedMask(
            binary_mask=np.zeros((height, width), dtype=np.uint8),
            area_ratio=0.0,
            bounding_box=(0, 0, 0, 0),
            discarded_region_count=0,
        )
