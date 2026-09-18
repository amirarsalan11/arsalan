"""
Pipeline Orchestrator: sequential execution of Preprocessor ->
Segmentor -> Mask Processor -> Geometry Estimator, with timing
metrics, early-exit short-circuiting on low-confidence/no-floor
results, and normalized error handling.

This module has zero knowledge of FastAPI, Celery, or the database —
per the architecture lock, "The AI Engine is independent from
FastAPI." It is a plain, importable Python class.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import cv2
import numpy as np

from ai_engine.exceptions import GeometryEstimationError, PipelineStageError
from ai_engine.geometry.geometry import GeometryEstimator
from ai_engine.mask_processing.mask_processor import MaskProcessor
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
    ) -> None:
        self._preprocessor = preprocessor if preprocessor is not None else Preprocessor()
        self._segmentor = segmentor if segmentor is not None else Segmentor()
        self._mask_processor = mask_processor if mask_processor is not None else MaskProcessor()
        self._geometry_estimator = (
            geometry_estimator if geometry_estimator is not None else GeometryEstimator()
        )

    def run(self, image_bytes: bytes) -> PipelineResult:
        """Run the full pipeline on raw image bytes.

        Raises:
            PipelineStageError (or a subclass): if the Preprocessor or
                Segmentor stage fails outright. Mask Processing and
                Geometry Estimation failures are handled more
                gracefully where possible (see module docstring on
                short-circuiting and geometry.py's degradation
                behavior) — but a truly degenerate mask can still
                raise GeometryEstimationError, which is caught here
                and converted into a `geometry_degraded` status rather
                than propagating, since a photo with no floor is a
                normal, expected outcome, not a pipeline bug.
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

        return PipelineResult(
            status=status,
            preprocessed=preprocessed,
            segmentation=segmentation,
            mask=mask,
            geometry=geometry,
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
