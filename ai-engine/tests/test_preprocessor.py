import io

import numpy as np
import pytest
from PIL import Image

from ai_engine.exceptions import ImageTooLargeError, InvalidImageError
from ai_engine.preprocessing.preprocessor import (
    MAX_SOURCE_DIMENSION,
    MIN_SOURCE_DIMENSION,
    MODEL_INPUT_SIZE,
    Preprocessor,
)
from tests.conftest import make_jpeg_bytes, make_png_bytes


@pytest.fixture
def preprocessor() -> Preprocessor:
    return Preprocessor()


def test_process_valid_jpeg_produces_correct_shapes(preprocessor: Preprocessor) -> None:
    result = preprocessor.process(make_jpeg_bytes(width=800, height=600))

    assert result.original_size == (800, 600)
    assert result.original_image.shape == (600, 800, 3)
    assert result.model_input.shape == (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3)
    assert result.model_input.dtype == np.uint8
    assert result.source_format == "JPEG"


def test_process_valid_png(preprocessor: Preprocessor) -> None:
    result = preprocessor.process(make_png_bytes(width=100, height=100))
    assert result.source_format == "PNG"
    assert result.original_size == (100, 100)


def test_letterbox_metadata_matches_aspect_ratio(preprocessor: Preprocessor) -> None:
    result = preprocessor.process(make_jpeg_bytes(width=800, height=600))
    # 800x600 -> scale = min(512/800, 512/600) = 0.64, landscape -> vertical padding
    assert abs(result.letterbox.scale - 0.64) < 1e-6
    assert result.letterbox.pad_left == 0
    assert result.letterbox.pad_right == 0
    assert result.letterbox.pad_top > 0
    assert result.letterbox.pad_bottom > 0


def test_letterbox_for_portrait_image_pads_horizontally(preprocessor: Preprocessor) -> None:
    result = preprocessor.process(make_jpeg_bytes(width=600, height=800))
    assert result.letterbox.pad_top == 0
    assert result.letterbox.pad_bottom == 0
    assert result.letterbox.pad_left > 0
    assert result.letterbox.pad_right > 0


def test_empty_bytes_raises_invalid_image_error(preprocessor: Preprocessor) -> None:
    with pytest.raises(InvalidImageError):
        preprocessor.process(b"")


def test_corrupt_bytes_raise_invalid_image_error(preprocessor: Preprocessor) -> None:
    with pytest.raises(InvalidImageError):
        preprocessor.process(b"this is definitely not an image")


def test_unsupported_format_raises_invalid_image_error(preprocessor: Preprocessor) -> None:
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="BMP")

    with pytest.raises(InvalidImageError):
        preprocessor.process(buf.getvalue())


def test_image_below_minimum_dimension_raises_invalid_image_error(
    preprocessor: Preprocessor,
) -> None:
    with pytest.raises(InvalidImageError):
        preprocessor.process(make_png_bytes(width=MIN_SOURCE_DIMENSION - 1, height=100))


def test_image_above_maximum_dimension_raises_image_too_large_error(
    preprocessor: Preprocessor,
) -> None:
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    # Resize the *header* expectation by directly constructing a large
    # PNG cheaply: a solid color large image compresses small on disk
    # but still reports full pixel dimensions to PIL.
    large = Image.new("RGB", (MAX_SOURCE_DIMENSION + 1, 100), color=(10, 10, 10))
    buf = io.BytesIO()
    large.save(buf, format="PNG")

    with pytest.raises(ImageTooLargeError):
        preprocessor.process(buf.getvalue())


def test_exif_orientation_is_applied_and_stripped(preprocessor: Preprocessor) -> None:
    # A tall (200x100 -> width=100,height=200) source, tagged with EXIF
    # orientation 6 (rotate 90 CW to display correctly), should result
    # in swapped width/height once corrected.
    arr = np.zeros((200, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    exif = img.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)

    result = preprocessor.process(buf.getvalue())
    assert result.original_size == (200, 100)


def test_rgba_image_is_converted_to_rgb(preprocessor: Preprocessor) -> None:
    arr = np.zeros((100, 100, 4), dtype=np.uint8)
    arr[..., 3] = 255
    img = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    result = preprocessor.process(buf.getvalue())
    assert result.original_image.shape == (100, 100, 3)


def test_process_valid_webp(preprocessor: Preprocessor) -> None:
    arr = (np.random.default_rng(10).random((150, 200, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="WEBP")

    result = preprocessor.process(buf.getvalue())
    assert result.source_format == "WEBP"
    assert result.original_size == (200, 150)
    assert result.original_image.shape == (150, 200, 3)


def test_grayscale_image_is_converted_to_three_channel_rgb(preprocessor: Preprocessor) -> None:
    gray_arr = (np.random.default_rng(11).random((150, 200)) * 255).astype(np.uint8)
    img = Image.fromarray(gray_arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    result = preprocessor.process(buf.getvalue())
    assert result.original_image.shape == (150, 200, 3)
    assert result.original_image.dtype == np.uint8
    # A true grayscale source should convert to R==G==B at every pixel.
    assert np.array_equal(result.original_image[..., 0], result.original_image[..., 1])
    assert np.array_equal(result.original_image[..., 1], result.original_image[..., 2])


def test_cmyk_image_is_converted_to_rgb(preprocessor: Preprocessor) -> None:
    cmyk_arr = (np.random.default_rng(12).random((150, 200, 4)) * 255).astype(np.uint8)
    img = Image.fromarray(cmyk_arr, mode="CMYK")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    result = preprocessor.process(buf.getvalue())
    assert result.original_image.shape == (150, 200, 3)
    assert result.original_image.dtype == np.uint8
