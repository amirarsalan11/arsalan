"""
Tests for material/material_processor.py.

All synthetic, in-memory images — no external fixture files. Covers
the scenarios required for this milestone: valid/invalid input,
format handling, grayscale/RGBA/transparency, aspect-ratio-preserving
normalization, determinism, oversized/undersized input, and no
uncontrolled upscaling.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from ai_engine.exceptions import ImageTooLargeError, InvalidImageError
from ai_engine.material.material_processor import (
    MAX_NORMALIZED_DIMENSION,
    MIN_TEXTURE_DIMENSION,
    MaterialProcessor,
)
from ai_engine.preprocessing.preprocessor import MAX_SOURCE_DIMENSION
from ai_engine.schemas import PreparedMaterial


@pytest.fixture
def material_processor() -> MaterialProcessor:
    return MaterialProcessor()


def _make_rgb_bytes(width: int, height: int, fmt: str = "PNG", color: int | tuple = 100) -> bytes:
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# --- valid input -----------------------------------------------------------


def test_valid_opaque_texture_produces_correct_output(material_processor: MaterialProcessor) -> None:
    data = _make_rgb_bytes(300, 200, fmt="JPEG")
    result = material_processor.process(data)

    assert isinstance(result, PreparedMaterial)
    assert result.texture_rgb.shape == (200, 300, 3)
    assert result.texture_rgb.dtype == np.uint8
    assert result.source_format == "JPEG"
    assert result.original_size == (300, 200)


def test_valid_png_texture(material_processor: MaterialProcessor) -> None:
    data = _make_rgb_bytes(150, 150, fmt="PNG")
    result = material_processor.process(data)
    assert result.source_format == "PNG"


def test_valid_webp_texture(material_processor: MaterialProcessor) -> None:
    arr = np.full((100, 120, 3), 80, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="WEBP")

    result = material_processor.process(buf.getvalue())
    assert result.source_format == "WEBP"
    assert result.original_size == (120, 100)


# --- invalid input ------------------------------------------------------


def test_empty_bytes_raises_invalid_image_error(material_processor: MaterialProcessor) -> None:
    with pytest.raises(InvalidImageError):
        material_processor.process(b"")


def test_corrupt_bytes_raise_invalid_image_error(material_processor: MaterialProcessor) -> None:
    with pytest.raises(InvalidImageError):
        material_processor.process(b"this is not a real image file")


def test_unsupported_format_raises_invalid_image_error(material_processor: MaterialProcessor) -> None:
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="BMP")

    with pytest.raises(InvalidImageError):
        material_processor.process(buf.getvalue())


def test_dimensions_below_minimum_raise_invalid_image_error(
    material_processor: MaterialProcessor,
) -> None:
    data = _make_rgb_bytes(MIN_TEXTURE_DIMENSION - 1, 50)
    with pytest.raises(InvalidImageError):
        material_processor.process(data)


def test_dimensions_above_maximum_raise_image_too_large_error(
    material_processor: MaterialProcessor,
) -> None:
    huge = Image.new("RGB", (MAX_SOURCE_DIMENSION + 1, 100), color=(10, 10, 10))
    buf = io.BytesIO()
    huge.save(buf, format="PNG")

    with pytest.raises(ImageTooLargeError):
        material_processor.process(buf.getvalue())


# --- grayscale / RGBA / transparency ------------------------------------


def test_grayscale_texture_is_converted_to_three_channel_rgb(
    material_processor: MaterialProcessor,
) -> None:
    gray_arr = (np.random.default_rng(20).random((100, 150)) * 255).astype(np.uint8)
    img = Image.fromarray(gray_arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    result = material_processor.process(buf.getvalue())
    assert result.texture_rgb.shape == (100, 150, 3)
    assert result.has_alpha is False
    assert np.all(result.alpha_mask == 255)
    # A true grayscale source converts to R==G==B at every pixel.
    assert np.array_equal(result.texture_rgb[..., 0], result.texture_rgb[..., 1])
    assert np.array_equal(result.texture_rgb[..., 1], result.texture_rgb[..., 2])


def test_rgba_texture_without_transparency_is_still_flagged_has_alpha(
    material_processor: MaterialProcessor,
) -> None:
    """An RGBA-mode source is has_alpha=True even if every pixel
    happens to be fully opaque — the flag reflects the source FORMAT
    carrying a channel, not whether transparency is actually used."""
    arr = np.zeros((100, 100, 4), dtype=np.uint8)
    arr[..., :3] = 50
    arr[..., 3] = 255
    img = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    result = material_processor.process(buf.getvalue())
    assert result.has_alpha is True
    assert np.all(result.alpha_mask == 255)


def test_rgba_transparency_is_preserved_not_discarded(
    material_processor: MaterialProcessor,
) -> None:
    arr = np.zeros((150, 150, 4), dtype=np.uint8)
    arr[..., :3] = 100
    arr[:75, :, 3] = 255  # top half opaque
    arr[75:, :, 3] = 0  # bottom half fully transparent
    img = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    result = material_processor.process(buf.getvalue())
    assert result.has_alpha is True
    assert result.alpha_mask[10, 10] == 255
    assert result.alpha_mask[140, 10] == 0


def test_partial_transparency_values_are_preserved(material_processor: MaterialProcessor) -> None:
    """Semi-transparent (not just 0/255) alpha values must survive
    intact — confirms the stage doesn't quantize/binarize alpha."""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[..., :3] = 100
    arr[..., 3] = 128  # exactly half-transparent everywhere
    img = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    result = material_processor.process(buf.getvalue())
    assert result.has_alpha is True
    assert np.all(result.alpha_mask == 128)


def test_opaque_source_format_synthesizes_full_opacity_alpha(
    material_processor: MaterialProcessor,
) -> None:
    data = _make_rgb_bytes(100, 100, fmt="JPEG")
    result = material_processor.process(data)
    assert result.has_alpha is False
    assert result.alpha_mask.shape == (100, 100)
    assert np.all(result.alpha_mask == 255)


# --- normalization: aspect ratio, determinism, no uncontrolled upscaling --


def test_aspect_ratio_is_preserved_when_downscaling(material_processor: MaterialProcessor) -> None:
    data = _make_rgb_bytes(2000, 1500, fmt="PNG")
    result = material_processor.process(data)

    assert result.was_downscaled is True
    orig_w, orig_h = result.original_size
    new_w, new_h = result.normalized_size
    assert abs((new_h / new_w) - (orig_h / orig_w)) < 0.01


def test_normalization_caps_at_max_normalized_dimension(
    material_processor: MaterialProcessor,
) -> None:
    data = _make_rgb_bytes(2000, 1500, fmt="PNG")
    result = material_processor.process(data)
    assert max(result.normalized_size) == MAX_NORMALIZED_DIMENSION


def test_small_texture_is_never_upscaled(material_processor: MaterialProcessor) -> None:
    data = _make_rgb_bytes(30, 20, fmt="PNG")
    result = material_processor.process(data)

    assert result.normalized_size == result.original_size == (30, 20)
    assert result.was_downscaled is False


def test_texture_under_normalization_cap_is_unchanged(
    material_processor: MaterialProcessor,
) -> None:
    data = _make_rgb_bytes(500, 400, fmt="PNG")
    result = material_processor.process(data)

    assert result.normalized_size == result.original_size == (500, 400)
    assert result.was_downscaled is False


def test_normalization_is_deterministic_across_repeated_calls(
    material_processor: MaterialProcessor,
) -> None:
    data = _make_rgb_bytes(2000, 1200, fmt="PNG", color=77)

    result_a = material_processor.process(data)
    result_b = material_processor.process(data)

    assert result_a.normalized_size == result_b.normalized_size
    assert np.array_equal(result_a.texture_rgb, result_b.texture_rgb)
    assert np.array_equal(result_a.alpha_mask, result_b.alpha_mask)


def test_alpha_and_rgb_are_resized_in_lockstep(material_processor: MaterialProcessor) -> None:
    """The alpha mask must end up at exactly the same normalized shape
    as the RGB texture, so a future consumer can index them together
    without any separate re-alignment step."""
    arr = np.zeros((2000, 1500, 4), dtype=np.uint8)
    arr[..., :3] = 90
    arr[:1000, :, 3] = 255
    arr[1000:, :, 3] = 0
    img = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    result = material_processor.process(buf.getvalue())
    assert result.texture_rgb.shape[:2] == result.alpha_mask.shape


# --- output schema shape ------------------------------------------------


def test_output_schema_fields_are_present_and_typed(material_processor: MaterialProcessor) -> None:
    data = _make_rgb_bytes(64, 64)
    result = material_processor.process(data)

    assert isinstance(result.texture_rgb, np.ndarray)
    assert isinstance(result.alpha_mask, np.ndarray)
    assert isinstance(result.has_alpha, bool)
    assert isinstance(result.original_size, tuple)
    assert isinstance(result.normalized_size, tuple)
    assert isinstance(result.source_format, str)
    assert isinstance(result.was_downscaled, bool)


def test_output_is_frozen(material_processor: MaterialProcessor) -> None:
    result = material_processor.process(_make_rgb_bytes(64, 64))
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        result.has_alpha = True  # type: ignore[misc]


# --- edge cases ----------------------------------------------------------


def test_minimum_valid_dimension_is_accepted(material_processor: MaterialProcessor) -> None:
    data = _make_rgb_bytes(MIN_TEXTURE_DIMENSION, MIN_TEXTURE_DIMENSION)
    result = material_processor.process(data)
    assert result.original_size == (MIN_TEXTURE_DIMENSION, MIN_TEXTURE_DIMENSION)


def test_square_texture_at_normalization_cap_boundary(
    material_processor: MaterialProcessor,
) -> None:
    data = _make_rgb_bytes(MAX_NORMALIZED_DIMENSION, MAX_NORMALIZED_DIMENSION)
    result = material_processor.process(data)
    assert result.was_downscaled is False
    assert result.normalized_size == (MAX_NORMALIZED_DIMENSION, MAX_NORMALIZED_DIMENSION)
