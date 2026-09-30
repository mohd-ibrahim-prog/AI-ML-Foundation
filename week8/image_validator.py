"""
image_validator.py

Purpose:
    First stage of the pipeline: is this even a readable, safe image
    file? This runs before any quality/plant/model checks.

    Everything here operates on in-memory bytes only. Nothing is ever
    written to disk, so there is no upload directory to secure, no
    filename to sanitize for storage, and no temporary file to clean
    up afterwards.

Week:
    AI/ML Foundation - Week 8
"""

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

import config


@dataclass
class FileValidationResult:
    ok: bool
    reason: str = ""
    image: Image.Image = None  # decoded RGB PIL image, if ok


def has_allowed_extension(filename):
    return Path(filename).suffix.lower() in config.ALLOWED_EXTENSIONS


def validate_file(filename, file_bytes):
    """
    Validate an uploaded file's name, size, and actual image content.

    Checks, in order:
    1. A filename was actually provided
    2. The extension is one we accept
    3. The file is not empty
    4. Pillow can decode it as an image at all (catches corrupted or
       non-image files pretending to have an image extension)
    5. The *decoded* format (JPEG/PNG) matches what we allow - this
       catches a file that was simply renamed to .jpg
    """

    if not filename:
        return FileValidationResult(False, "No file was provided.")

    if not has_allowed_extension(filename):
        return FileValidationResult(
            False, "Unsupported file type. Please upload a JPG or PNG image."
        )

    if not file_bytes:
        return FileValidationResult(False, "The uploaded file is empty.")

    try:
        probe = Image.open(io.BytesIO(file_bytes))
        probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return FileValidationResult(
            False, "This file could not be read as an image. It may be corrupted."
        )

    try:
        # verify() invalidates the file handle - reopen for real use.
        image = Image.open(io.BytesIO(file_bytes))
        detected_format = image.format
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return FileValidationResult(
            False, "This file could not be read as an image. It may be corrupted."
        )

    if detected_format not in config.ALLOWED_PIL_FORMATS:
        return FileValidationResult(
            False,
            "The file's actual content does not match a supported image "
            "format (JPG/PNG).",
        )

    return FileValidationResult(True, image=image)
