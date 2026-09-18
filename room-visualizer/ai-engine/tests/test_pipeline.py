"""
Tests for pipeline.py (RenderPipeline).

Every test injects a FakeSegmentationBackend via Segmentor, so no real
model weights are ever loaded. Real Preprocessor, MaskProcessor, and
GeometryEstimator instances run for real against synthetic images —
only the segmentation model itself is faked.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai_engine.exceptions import InvalidImageError, PipelineStageError
from ai_engine.pipeline import RenderPipeline
from ai_engine.segmentation.segmentor import Segmentor
from tests.conftest import (
    FakeSegmentationBackend,
    make_jpeg_bytes,
    make_trapezoid_room_jpeg,
)


def _pipeline_with_mask(mask: np.ndarray) -> RenderPipeline:
    return RenderPipeline(segmentor=Segmentor(backend=FakeSegmentationBackend(mask)))


def test_full_pipeline_reaches_ok_status_with_confident_floor(confident_floor_mask) -> None:
    pipeline = _pipeline_with_mask(confident_floor_mask)
    result = pipeline.run(make_trapezoid_room_jpeg())

    assert result.status in ("ok", "geometry_degraded")
    assert result.geometry is not None
    assert result.mask.area_ratio > 0.0


def test_pipeline_records_timing_for_every_stage_that_ran(confident_floor_mask) -> None:
    pipeline = _pipeline_with_mask(confident_floor_mask)
    result = pipeline.run(make_trapezoid_room_jpeg())

    stage_names = [t.stage_name for t in result.timings]
    assert stage_names == ["preprocessing", "segmentation", "mask_processing", "geometry"]
    for timing in result.timings:
        assert timing.duration_ms >= 0.0


def test_low_confidence_short_circuits_before_mask_and_geometry(low_confidence_mask) -> None:
    pipeline = _pipeline_with_mask(low_confidence_mask)
    result = pipeline.run(make_jpeg_bytes())

    assert result.status == "no_floor_detected"
    assert result.geometry is None
    stage_names = [t.stage_name for t in result.timings]
    assert stage_names == ["preprocessing", "segmentation"]
    assert any(w.code == "LOW_SEGMENTATION_CONFIDENCE" for w in result.warnings)


def test_insufficient_area_short_circuits_before_geometry(tiny_floor_mask) -> None:
    pipeline = _pipeline_with_mask(tiny_floor_mask)
    result = pipeline.run(make_jpeg_bytes())

    assert result.status == "no_floor_detected"
    assert result.geometry is None
    stage_names = [t.stage_name for t in result.timings]
    assert stage_names == ["preprocessing", "segmentation", "mask_processing"]
    assert any(w.code == "INSUFFICIENT_FLOOR_AREA" for w in result.warnings)


def test_invalid_image_raises_structured_error() -> None:
    pipeline = _pipeline_with_mask(np.zeros((512, 512), dtype=np.float32))

    with pytest.raises(InvalidImageError):
        pipeline.run(b"not a real image")


def test_invalid_image_error_is_a_pipeline_stage_error() -> None:
    """Confirms the exception hierarchy: a caller catching the base
    PipelineStageError catches this too, without needing to know the
    specific subclass."""
    pipeline = _pipeline_with_mask(np.zeros((512, 512), dtype=np.float32))

    with pytest.raises(PipelineStageError):
        pipeline.run(b"not a real image")


def test_geometry_degraded_status_when_vanishing_point_unresolved() -> None:
    # A confident, roughly-rectangular floor mask with no drawn lines
    # at all in the image -> geometry succeeds but can't find a
    # vanishing point, and must degrade rather than raise/propagate.
    mask = np.zeros((512, 512), dtype=np.float32)
    mask[150:480, 60:460] = 0.9

    pipeline = _pipeline_with_mask(mask)
    result = pipeline.run(make_jpeg_bytes(width=640, height=480, color=(180, 180, 180)))

    assert result.status == "geometry_degraded"
    assert result.geometry is not None
    assert result.geometry.vanishing_point is None
    assert any(w.code == "VANISHING_POINT_UNRESOLVED" for w in result.warnings)


def test_multiple_regions_warning_surfaces_on_pipeline_result(confident_floor_mask) -> None:
    mask = confident_floor_mask.copy()
    # confident_floor_mask's main blob occupies rows 150:480, cols 60:460.
    # make_trapezoid_room_jpeg() defaults to 640x480, whose letterbox
    # crops raw-mask rows to 64:448 (padding outside that is discarded
    # before mask processing ever sees it) — so the extra disconnected
    # blob must sit inside that surviving band, away from the main blob.
    mask[70:90, 470:495] = 0.9

    pipeline = _pipeline_with_mask(mask)
    result = pipeline.run(make_trapezoid_room_jpeg())

    assert any(w.code == "MULTIPLE_REGIONS_FOUND" for w in result.warnings)


def test_pipeline_default_construction_uses_real_stage_instances() -> None:
    from ai_engine.geometry.geometry import GeometryEstimator
    from ai_engine.mask_processing.mask_processor import MaskProcessor
    from ai_engine.preprocessing.preprocessor import Preprocessor

    pipeline = RenderPipeline()
    assert isinstance(pipeline._preprocessor, Preprocessor)
    assert isinstance(pipeline._segmentor, Segmentor)
    assert isinstance(pipeline._mask_processor, MaskProcessor)
    assert isinstance(pipeline._geometry_estimator, GeometryEstimator)


def test_pipeline_accepts_injected_stage_instances(confident_floor_mask) -> None:
    from ai_engine.geometry.geometry import GeometryEstimator
    from ai_engine.mask_processing.mask_processor import MaskProcessor
    from ai_engine.preprocessing.preprocessor import Preprocessor

    custom_preprocessor = Preprocessor()
    custom_mask_processor = MaskProcessor()
    custom_geometry = GeometryEstimator()
    custom_segmentor = Segmentor(backend=FakeSegmentationBackend(confident_floor_mask))

    pipeline = RenderPipeline(
        preprocessor=custom_preprocessor,
        segmentor=custom_segmentor,
        mask_processor=custom_mask_processor,
        geometry_estimator=custom_geometry,
    )

    assert pipeline._preprocessor is custom_preprocessor
    assert pipeline._segmentor is custom_segmentor
    assert pipeline._mask_processor is custom_mask_processor
    assert pipeline._geometry_estimator is custom_geometry


def test_unexpected_stage_exception_is_wrapped_as_pipeline_stage_error(confident_floor_mask) -> None:
    """An unexpected (non-PipelineStageError) exception raised inside
    a stage must be normalized to PipelineStageError, not leak the
    original exception type to the caller."""

    class ExplodingSegmentor:
        def segment(self, model_input):
            raise RuntimeError("something totally unexpected broke")

    pipeline = RenderPipeline(segmentor=ExplodingSegmentor())

    with pytest.raises(PipelineStageError) as exc_info:
        pipeline.run(make_jpeg_bytes())

    assert exc_info.value.stage == "segmentation"
    assert isinstance(exc_info.value.original_exception, RuntimeError)
