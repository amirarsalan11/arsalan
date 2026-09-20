"""
Tests for segmentation/model_registry.py and segmentation/segmentor.py.

No real model weights are ever downloaded or required. Segmentor's own
logic is tested entirely against `FakeSegmentationBackend` (satisfying
the `SegmentationBackend` protocol). The model registry's *loading*
behavior (singleton caching, thread-safety, reset) is tested by
monkeypatching `SegmentationModelRegistry._load` with a lightweight
fake loader — never by actually loading transformers/torch weights.

The one place this suite genuinely exercises the real, un-mocked load
path is `test_registry_raises_model_load_error_when_dependencies_missing`
— which is a meaningful test specifically because torch/transformers
are not installed in the environment this was validated in, and the
resulting ImportError-to-ModelLoadError translation is real production
code being exercised, not a mock.
"""

from __future__ import annotations

import sys
import threading

import numpy as np
import pytest

from ai_engine.exceptions import InferenceError, ModelLoadError
from ai_engine.segmentation.model_registry import (
    SegmentationModelBundle,
    SegmentationModelRegistry,
)
from ai_engine.segmentation.segmentor import Segmentor, TorchSegmentationBackend
from tests.conftest import FakeSegmentationBackend


@pytest.fixture(autouse=True)
def reset_registry():
    """Every test starts and ends with a clean registry, so tests
    can't leak a cached bundle (real or fake) into each other."""
    SegmentationModelRegistry.reset()
    yield
    SegmentationModelRegistry.reset()


# --- model_registry.py: singleton / loading behavior -----------------------


def test_registry_loads_once_across_repeated_calls(monkeypatch) -> None:
    load_call_count = 0

    def fake_load(model_identifier: str) -> SegmentationModelBundle:
        nonlocal load_call_count
        load_call_count += 1
        return SegmentationModelBundle(model=object(), image_processor=object(), model_version="fake-v1")

    monkeypatch.setattr(SegmentationModelRegistry, "_load", staticmethod(fake_load))

    first = SegmentationModelRegistry.get("some/identifier")
    second = SegmentationModelRegistry.get("some/identifier")

    assert load_call_count == 1
    assert first is second


def test_registry_reset_forces_reload(monkeypatch) -> None:
    load_call_count = 0

    def fake_load(model_identifier: str) -> SegmentationModelBundle:
        nonlocal load_call_count
        load_call_count += 1
        return SegmentationModelBundle(model=object(), image_processor=object(), model_version="fake-v1")

    monkeypatch.setattr(SegmentationModelRegistry, "_load", staticmethod(fake_load))

    SegmentationModelRegistry.get()
    SegmentationModelRegistry.reset()
    SegmentationModelRegistry.get()

    assert load_call_count == 2


def test_registry_is_thread_safe_under_concurrent_first_load(monkeypatch) -> None:
    load_call_count = 0
    load_lock = threading.Lock()

    def fake_slow_load(model_identifier: str) -> SegmentationModelBundle:
        nonlocal load_call_count
        # Simulate a slow load so concurrent callers actually race.
        import time

        time.sleep(0.05)
        with load_lock:
            load_call_count += 1
        return SegmentationModelBundle(model=object(), image_processor=object(), model_version="fake-v1")

    monkeypatch.setattr(SegmentationModelRegistry, "_load", staticmethod(fake_slow_load))

    results: list[SegmentationModelBundle] = []
    results_lock = threading.Lock()

    def worker():
        bundle = SegmentationModelRegistry.get()
        with results_lock:
            results.append(bundle)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert load_call_count == 1, "model should be loaded exactly once despite concurrent callers"
    assert len(results) == 8
    assert all(bundle is results[0] for bundle in results)


def test_registry_raises_model_load_error_when_dependencies_missing() -> None:
    """Real (unmocked) load path. Meaningful specifically because
    torch/transformers are genuinely not installed in this
    environment — confirms the ImportError -> ModelLoadError
    translation actually works, not just that it's written."""
    with pytest.raises(ModelLoadError):
        SegmentationModelRegistry.get()


# --- segmentor.py: TorchSegmentationBackend without torch installed --------


def test_torch_backend_raises_inference_error_without_torch(monkeypatch) -> None:
    """Deterministically forces `import torch` to fail inside
    `predict()`, regardless of whether torch is actually installed in
    whatever environment runs this test.

    Root-cause note: this test previously relied on torch genuinely
    being ABSENT from the ambient environment to exercise predict()'s
    `except ImportError` branch. That happened to hold in one sandbox
    but is not true of a normal development machine with the AI
    dependency stack installed — there, `import torch` succeeds and
    execution reaches `SegmentationModelRegistry.get()` instead, which
    correctly raises ModelLoadError (not InferenceError) once
    Hugging Face's `from_pretrained()` fails to find the placeholder
    checkpoint. That is correct production behavior, not a bug — see
    `test_torch_backend_propagates_model_load_error_when_registry_load_fails`
    below for that path. This test's OWN job is narrower: prove that
    IF torch itself is unavailable, predict() raises InferenceError.
    Setting `sys.modules["torch"] = None` is the standard technique
    for forcing that precise precondition deterministically — CPython
    raises ImportError immediately for any `import torch` while that
    entry is set to None — so this test passes identically whether or
    not torch is actually installed, while still exercising the real,
    unmocked `except ImportError` line in predict().
    """
    monkeypatch.setitem(sys.modules, "torch", None)

    backend = TorchSegmentationBackend()
    with pytest.raises(InferenceError):
        backend.predict(np.zeros((512, 512, 3), dtype=np.uint8))


def test_torch_backend_propagates_model_load_error_when_registry_load_fails(monkeypatch) -> None:
    """Covers the scenario a real, torch-installed environment
    actually hits: torch imports fine, but the model registry's load
    fails (e.g. Hugging Face returns 401/RepositoryNotFoundError for
    the placeholder checkpoint identifier). predict() must let
    ModelLoadError propagate uncaught — NOT re-categorize it as
    InferenceError — since a load failure and an inference failure
    are deliberately distinct in this project's exception hierarchy.

    A minimal fake `torch` module is injected into sys.modules so
    `import torch` succeeds deterministically regardless of whether
    real torch is actually installed wherever this test runs — the
    fake module never needs any real behavior, since `_load` is made
    to fail before predict() ever touches a torch attribute. `_load`
    itself is monkeypatched (same established pattern as the registry
    tests above); no network access occurs either way.
    """
    import types

    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    def failing_load(model_identifier: str) -> SegmentationModelBundle:
        raise ModelLoadError(
            "model_registry", f"Simulated 401/RepositoryNotFoundError for '{model_identifier}'."
        )

    monkeypatch.setattr(SegmentationModelRegistry, "_load", staticmethod(failing_load))

    backend = TorchSegmentationBackend()
    with pytest.raises(ModelLoadError):
        backend.predict(np.zeros((512, 512, 3), dtype=np.uint8))


def test_floor_class_index_is_configurable_per_instance() -> None:
    default_backend = TorchSegmentationBackend()
    overridden_backend = TorchSegmentationBackend(floor_class_index=7)

    assert default_backend._floor_class_index == 3
    assert overridden_backend._floor_class_index == 7


# --- segmentor.py: Segmentor orchestration via FakeSegmentationBackend -----


def test_segmentor_output_shape_and_range(confident_floor_mask) -> None:
    segmentor = Segmentor(backend=FakeSegmentationBackend(confident_floor_mask))
    result = segmentor.segment(np.zeros((512, 512, 3), dtype=np.uint8))

    assert result.raw_mask.shape == (512, 512)
    assert result.raw_mask.dtype == np.float32
    assert result.raw_mask.min() >= 0.0
    assert result.raw_mask.max() <= 1.0


def test_segmentor_confidence_is_mean_over_floor_pixels_only() -> None:
    mask = np.zeros((512, 512), dtype=np.float32)
    mask[100:400, 100:400] = 0.85  # confident region
    # Everything else is 0.0 (confidently non-floor) — must NOT drag
    # down the reported confidence, which should reflect only the
    # predicted-floor region.
    segmentor = Segmentor(backend=FakeSegmentationBackend(mask))
    result = segmentor.segment(np.zeros((512, 512, 3), dtype=np.uint8))

    assert abs(result.mean_confidence - 0.85) < 1e-4


def test_segmentor_confidence_is_zero_when_no_floor_predicted() -> None:
    mask = np.zeros((512, 512), dtype=np.float32)
    segmentor = Segmentor(backend=FakeSegmentationBackend(mask))
    result = segmentor.segment(np.zeros((512, 512, 3), dtype=np.uint8))

    assert result.mean_confidence == 0.0


def test_segmentor_preserves_model_version(confident_floor_mask) -> None:
    segmentor = Segmentor(backend=FakeSegmentationBackend(confident_floor_mask, version="my-checkpoint-v2"))
    result = segmentor.segment(np.zeros((512, 512, 3), dtype=np.uint8))

    assert result.model_version == "my-checkpoint-v2"


def test_segmentor_default_backend_is_torch_backend() -> None:
    segmentor = Segmentor()
    assert isinstance(segmentor._backend, TorchSegmentationBackend)
