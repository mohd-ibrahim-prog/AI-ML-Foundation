"""Tests for image_validator.py - file-level validation."""

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import image_validator as iv


def _valid_jpeg_bytes():
    img = Image.fromarray((np.random.default_rng(0).random((80, 80, 3)) * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


def test_missing_filename_is_rejected():
    result = iv.validate_file("", _valid_jpeg_bytes())
    assert result.ok is False


def test_disallowed_extension_is_rejected():
    result = iv.validate_file("notes.txt", b"hello world")
    assert result.ok is False
    assert "unsupported file type" in result.reason.lower()


def test_empty_file_is_rejected():
    result = iv.validate_file("photo.jpg", b"")
    assert result.ok is False


def test_corrupted_file_is_rejected():
    result = iv.validate_file("photo.jpg", b"this is not a real image" * 10)
    assert result.ok is False
    assert "corrupted" in result.reason.lower() or "could not be read" in result.reason.lower()


def test_truncated_jpeg_is_rejected():
    data = _valid_jpeg_bytes()
    result = iv.validate_file("photo.jpg", data[: len(data) // 2])
    assert result.ok is False


def test_renamed_file_with_mismatched_format_is_rejected():
    # A valid PNG, saved with a .jpg extension - the decoded format
    # (PNG) won't match the allowed set the way a real JPEG would.
    img = Image.fromarray((np.random.default_rng(1).random((80, 80, 3)) * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, "BMP")  # a format we do not allow at all
    result = iv.validate_file("photo.jpg", buf.getvalue())
    assert result.ok is False


def test_valid_jpeg_passes():
    result = iv.validate_file("photo.jpg", _valid_jpeg_bytes())
    assert result.ok is True
    assert result.image is not None
