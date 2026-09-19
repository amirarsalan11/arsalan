"""
Shared test fixtures and synthetic-image helpers for the AI pipeline
test suite. No real photographs are used — every test builds its own
minimal synthetic image/mask, keeping tests fast, deterministic, and
independent of any external test-data files.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image


def make_jpeg_bytes(width: int = 640, height: int = 480, color: tuple[int, int, int] | None = None) -> bytes:
    if color is not None:
        arr = np.full((height, width, 3), color, dtype=np.uint8)
    else:
        arr = (np.random.default_rng(0).random((height, width, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_png_bytes(width: int = 100, height: int = 100) -> bytes:
    arr = (np.random.default_rng(1).random((height, width, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_trapezoid_room_jpeg(width: int = 640, height: int = 480) -> bytes:
    """A synthetic "room" with a trapezoidal floor and real converging
    edge lines drawn into the pixel data — used by geometry/pipeline
    tests that need actual Hough-detectable lines, not just a mask."""
    arr = np.full((height, width, 3), 200, dtype=np.uint8)
    trapezoid = np.array(
        [[80, height - 10], [560, height - 10], [420, 220], [220, 220]], dtype=np.int32
    )
    cv2.fillPoly(arr, [trapezoid], (150, 120, 90))
    cv2.line(arr, (80, height - 10), (220, 220), (60, 60, 60), 3)
    cv2.line(arr, (560, height - 10), (420, 220), (60, 60, 60), 3)
    cv2.line(arr, (120, height - 10), (235, 220), (60, 60, 60), 2)
    cv2.line(arr, (520, height - 10), (405, 220), (60, 60, 60), 2)

    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class FakeSegmentationBackend:
    """Test double satisfying segmentor.SegmentationBackend, used to
    exercise Segmentor/RenderPipeline without requiring torch."""

    def __init__(self, mask: np.ndarray, version: str = "fake-v1") -> None:
        self._mask = mask
        self._version = version

    def predict(self, model_input: np.ndarray) -> tuple[np.ndarray, str]:
        return self._mask, self._version


@pytest.fixture
def confident_floor_mask() -> np.ndarray:
    mask = np.zeros((512, 512), dtype=np.float32)
    mask[150:480, 60:460] = 0.9
    return mask


@pytest.fixture
def low_confidence_mask() -> np.ndarray:
    return np.full((512, 512), 0.1, dtype=np.float32)


@pytest.fixture
def tiny_floor_mask() -> np.ndarray:
    mask = np.zeros((512, 512), dtype=np.float32)
    mask[250:260, 250:260] = 0.9
    return mask
