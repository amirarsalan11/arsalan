"""
Model registry: lazy-loaded, singleton access to the segmentation
model's weights.

This module exists specifically to satisfy the architecture rule
"Load model once. Never load model per request. Separate model
loading from inference." — segmentor.py's inference code calls into
this registry rather than ever constructing a model itself.
"""

from __future__ import annotations

import threading

from ai_engine.exceptions import ModelLoadError

# Default checkpoint. A fine-tuned floor-segmentation SegFormer-B2
# checkpoint is assumed to exist at this Hugging Face Hub identifier
# (or a local path override, see `SegmentationModelRegistry.get`) —
# actually publishing/training that checkpoint is outside this
# milestone's scope; this registry only handles the loading contract.
DEFAULT_MODEL_IDENTIFIER = "room-visualizer/segformer-b2-floor-segmentation"


class SegmentationModelBundle:
    """Holds a loaded model + its matching image processor together,
    so callers never accidentally pair mismatched versions of each."""

    def __init__(self, model: object, image_processor: object, model_version: str) -> None:
        self.model = model
        self.image_processor = image_processor
        self.model_version = model_version


class SegmentationModelRegistry:
    """Process-wide singleton. Thread-safe lazy initialization: the
    first caller (from any thread) triggers the actual load; every
    other caller, on any thread, reuses the same loaded bundle.
    """

    _bundle: SegmentationModelBundle | None = None
    _bundle_lock = threading.Lock()

    @classmethod
    def get(cls, model_identifier: str = DEFAULT_MODEL_IDENTIFIER) -> SegmentationModelBundle:
        """Return the loaded model bundle, loading it on first call.

        Raises:
            ModelLoadError: if the underlying weights/processor fail
                to load (missing files, corrupt checkpoint, missing
                torch/transformers installation, etc). Fails fast and
                loudly rather than surfacing as a mysterious failure
                deep inside a request.
        """
        if cls._bundle is not None:
            return cls._bundle

        with cls._bundle_lock:
            # Re-check inside the lock: another thread may have
            # finished loading while this thread was waiting for it.
            if cls._bundle is not None:
                return cls._bundle

            cls._bundle = cls._load(model_identifier)
            return cls._bundle

    @classmethod
    def reset(cls) -> None:
        """Discard the currently loaded bundle, forcing the next
        `get()` call to reload from scratch. Intended for tests and
        for an explicit, deliberate model-version rotation — never
        called as part of normal request handling."""
        with cls._bundle_lock:
            cls._bundle = None

    @classmethod
    def _load(cls, model_identifier: str) -> SegmentationModelBundle:
        try:
            # Imported lazily, inside the load path, rather than at
            # module level: torch/transformers are heavy, optional-at-
            # import-time dependencies, and importing them only when a
            # model is actually requested keeps every other module in
            # this package (preprocessor, mask_processor, geometry)
            # usable in environments where the full AI stack isn't
            # installed (e.g. this milestone's own test/validation
            # environment for the non-model stages).
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError as exc:
            raise ModelLoadError(
                "model_registry",
                "PyTorch/transformers are not installed. The segmentation "
                "stage requires the full AI dependency stack.",
                exc,
            ) from exc

        try:
            model = SegformerForSemanticSegmentation.from_pretrained(model_identifier)
            model.eval()
            image_processor = SegformerImageProcessor.from_pretrained(model_identifier)
        except Exception as exc:  # noqa: BLE001 - any load failure must become ModelLoadError
            raise ModelLoadError(
                "model_registry",
                f"Failed to load segmentation model '{model_identifier}'.",
                exc,
            ) from exc

        return SegmentationModelBundle(
            model=model, image_processor=image_processor, model_version=model_identifier
        )
