"""
dataset_utils.py

Purpose:
    Locate the PlantVillage dataset on disk and discover the
    available image files and their disease/class labels.

    The dataset itself is NOT stored inside week5/. It is reused
    from wherever it already lives on the user's machine (for
    example, the copy downloaded for Week 4), so the dataset is
    never duplicated.

Week:
    AI/ML Foundation - Week 5
"""

import os
import re
from pathlib import Path

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Root of the repository (AIML FOUNDATION/)
REPO_ROOT = Path(__file__).resolve().parents[2]
WEEK5_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# DATASET LOCATION
# ---------------------------------------------------------

def resolve_dataset_dir():
    """
    Find the PlantVillage dataset folder.

    Checks, in order:
    1. The PLANTVILLAGE_DATA_DIR environment variable
       (lets the user point to any location, e.g. on Windows:
        set PLANTVILLAGE_DATA_DIR=C:\\path\\to\\PlantVillage)
    2. week5/data/raw/PlantVillage (a local copy, if the user
       placed one here)
    3. week4/data/raw/PlantVillage (the copy already used in
       Week 4, reused instead of duplicated)

    Raises:
        FileNotFoundError with a clear, actionable message if the
        dataset cannot be found in any of these locations.
    """

    candidates = []

    env_path = os.environ.get("PLANTVILLAGE_DATA_DIR")
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(WEEK5_DIR / "data" / "raw" / "PlantVillage")
    candidates.append(REPO_ROOT / "week4" / "data" / "raw" / "PlantVillage")

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    checked = "\n".join(f"  - {c}" for c in candidates)

    raise FileNotFoundError(
        "Could not locate the PlantVillage dataset.\n\n"
        "Checked the following locations:\n"
        f"{checked}\n\n"
        "To fix this, do ONE of the following:\n"
        "  1. Reuse the dataset already downloaded for Week 4 "
        "(recommended) - no action needed if it exists there.\n"
        "  2. Copy/extract the PlantVillage 'color' folder into:\n"
        f"     {WEEK5_DIR / 'data' / 'raw' / 'PlantVillage'}\n"
        "  3. Set an environment variable pointing to your dataset:\n"
        "     Windows PowerShell:\n"
        "       $env:PLANTVILLAGE_DATA_DIR = 'C:\\path\\to\\PlantVillage'\n"
        "     Linux / macOS:\n"
        "       export PLANTVILLAGE_DATA_DIR=/path/to/PlantVillage\n"
    )


# ---------------------------------------------------------
# LABEL EXTRACTION
# ---------------------------------------------------------

def extract_label(file_path):
    """
    Extract the disease/class label for an image.

    Supports:
    1. Folder-based labels (PlantVillage 'color' layout:
       one sub-folder per class)
    2. Filename-based labels, as a fallback, using the same
       convention introduced in Week 4:
       abc123__FREC_Scab 3112.JPG -> FREC_Scab
    """

    file_path = Path(file_path)
    parent_name = file_path.parent.name

    if parent_name != "PlantVillage":
        return parent_name

    filename = file_path.stem

    if "__" in filename:
        label_part = filename.split("__", 1)[1]
        label_part = re.sub(r"\s+\d+$", "", label_part)
        return label_part.strip()

    return "Unknown"


# ---------------------------------------------------------
# DISCOVER IMAGES
# ---------------------------------------------------------

def discover_images_and_labels(dataset_dir):
    """
    Walk the dataset directory and collect image file paths and
    their corresponding labels.

    File order is sorted so that the resulting list (and any
    train/validation/test split built from it with a fixed seed)
    is reproducible across runs and across weeks.

    Returns:
        image_files (list[Path]), labels (list[str]), class_names (list[str])
    """

    image_files = sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        raise ValueError(f"No image files found inside: {dataset_dir}")

    labels = [extract_label(path) for path in image_files]
    class_names = sorted(set(labels))

    return image_files, labels, class_names


def print_dataset_summary(dataset_dir, image_files, labels, class_names):
    """Print a short, human-readable summary of the discovered dataset."""

    print(f"\nDataset location:")
    print(dataset_dir)

    print(f"\nTotal images found   : {len(image_files)}")
    print(f"Total disease classes: {len(class_names)}")

    print("\nDisease classes:")
    for index, label in enumerate(class_names, start=1):
        count = labels.count(label)
        print(f"{index:2}. {label:<35} {count} images")
