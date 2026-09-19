"""
Segmentor: runs the floor-segmentation model on a 512x512 model-input
array and produces a per-pixel floor probability mask.

Backend abstraction: `SegmentationBackend` is a narrow protocol with
one method, `predict`. `TorchSegmentationBackend` is the only
implementation in this milestone. This satisfies "keep architecture
ready for ONNX Runtime migration" (a future `ONNXSegmentationBackend`
could implement the same protocol) without actually building ONNX
support now, per "do not implement optimization prematurely."

Model loading is intentionally NOT done here — see model_registry.py.
This module only performs inference against an already-loaded model.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ai_engine.exceptions import InferenceError
from ai_engine.schemas import SegmentationResult
from ai_engine.segmentation.model_registry import SegmentationModelRegistry

# --------------------------------------------------------------------------
# IMPORTANT MODEL/CHECKPOINT ASSUMPTION — READ BEFORE CHANGING THE CHECKPOINT
# --------------------------------------------------------------------------
# FLOOR_CLASS_INDEX is the index of the "floor" class within the
# segmentation model's output class dimension. This is NOT a property
# of this code — it is a property of whatever specific checkpoint
# `model_registry.DEFAULT_MODEL_IDENTIFIER` resolves to, determined
# entirely by that checkpoint's own label/id2label mapping at training
# time.
#
# The value below (3) is a PLACEHOLDER. No real fine-tuned floor-
# segmentation checkpoint has been selected or verified against this
# codebase as of this milestone — model training/selection is outside
# this milestone's scope. Whoever selects the production checkpoint
# MUST:
#   1. Inspect that checkpoint's `config.id2label` mapping directly.
#   2. Confirm which integer index corresponds to "floor" (or whatever
#      the trained label is actually named).
#   3. Update FLOOR_CLASS_INDEX (or pass `floor_class_index=...`
#      explicitly to TorchSegmentationBackend — see below) accordingly.
#
# Do NOT assume this value transfers to a different checkpoint without
# doing that verification — silently reusing it would misroute an
# unrelated class's probabilities as "floor" with no error raised
# anywhere, since every downstream shape/range check would still pass.
FLOOR_CLASS_INDEX = 3


class SegmentationBackend(Protocol):
    """Narrow interface a segmentation backend must implement.
    `TorchSegmentationBackend` is the only implementation today; a
    future `ONNXSegmentationBackend` would implement this same
    protocol without requiring any change to Segmentor itself."""

    def predict(self, model_input: np.ndarray) -> tuple[np.ndarray, str]:
        """Return (floor_probability_mask, model_version).

        `floor_probability_mask` must be a 512x512 float32 array with
        values in [0, 1].
        """
        ...


class TorchSegmentationBackend:
    """PyTorch/HuggingFace Transformers implementation of
    SegmentationBackend, backed by the process-wide model registry.

    `floor_class_index` defaults to the module-level FLOOR_CLASS_INDEX
    placeholder but can be overridden per-instance — e.g. once a real
    checkpoint is selected and its actual floor-class index is
    confirmed, without needing to edit this file.
    """

    def __init__(
        self,
        model_identifier: str | None = None,
        floor_class_index: int = FLOOR_CLASS_INDEX,
    ) -> None:
        self._model_identifier = model_identifier
        self._floor_class_index = floor_class_index

    def predict(self, model_input: np.ndarray) -> tuple[np.ndarray, str]:
        try:
            import torch
        except ImportError as exc:
            raise InferenceError(
                "segmentor", "PyTorch is not installed; cannot run inference.", exc
            ) from exc

        if self._model_identifier is not None:
            bundle = SegmentationModelRegistry.get(self._model_identifier)
        else:
            bundle = SegmentationModelRegistry.get()

        try:
            inputs = bundle.image_processor(images=model_input, return_tensors="pt")

            with torch.no_grad():
                outputs = bundle.model(**inputs)
                logits = outputs.logits  # shape: (1, num_classes, H', W'), H'/W' < 512 typically

                # Upsample to the full 512x512 model-input resolution
                # BEFORE any per-pixel decision is made, rather than
                # thresholding at the decoder's native (lower)
                # resolution and upscaling a hard mask afterward.
                upsampled_logits = torch.nn.functional.interpolate(
                    logits,
                    size=model_input.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )

                probabilities = torch.softmax(upsampled_logits, dim=1)
                floor_probabilities = probabilities[0, self._floor_class_index, :, :]

            floor_mask = floor_probabilities.cpu().numpy().astype(np.float32)
        except InferenceError:
            raise
        except Exception as exc:  # noqa: BLE001 - any inference failure must become InferenceError
            raise InferenceError(
                "segmentor", "Segmentation model inference failed.", exc
            ) from exc

        return floor_mask, bundle.model_version


class Segmentor:
    """Stateless orchestration around a SegmentationBackend — safe to
    reuse a single instance across requests. The backend itself may
    hold a reference to the (singleton, cached) loaded model."""

    def __init__(self, backend: SegmentationBackend | None = None) -> None:
        self._backend = backend if backend is not None else TorchSegmentationBackend()

    def segment(self, model_input: np.ndarray) -> SegmentationResult:
        """
        Args:
            model_input: 512x512x3 uint8 RGB array (PreprocessedImage.model_input).

        Raises:
            InferenceError: if the model fails to load or the forward
                pass fails.
        """
        floor_mask, model_version = self._backend.predict(model_input)

        # Confidence is the mean predicted probability specifically
        # within the region the model itself believes is floor, not
        # the mean over the whole frame — a small, highly-confident
        # floor region shouldn't be penalized just because most of the
        # frame is (correctly) predicted as non-floor.
        floor_pixels = floor_mask[floor_mask >= 0.5]
        mean_confidence = float(floor_pixels.mean()) if floor_pixels.size > 0 else 0.0

        return SegmentationResult(
            raw_mask=floor_mask,
            mean_confidence=mean_confidence,
            model_version=model_version,
        )
