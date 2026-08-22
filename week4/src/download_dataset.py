"""
Week 4 - PlantVillage Dataset Downloader
"""

import subprocess
from pathlib import Path


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

WEEK4_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WEEK4_DIR / "data" / "raw"
REPO_DIR = DATA_DIR / "PlantVillage-Dataset"


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("PLANTVILLAGE DATASET SETUP")
    print("=" * 70)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 1. CLONE REPOSITORY
    # -----------------------------------------------------

    if not REPO_DIR.exists():

        print("\nDownloading PlantVillage repository...")
        print("This may take some time.")

        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/spMohanty/PlantVillage-Dataset.git",
                str(REPO_DIR)
            ],
            check=True
        )

        print("\nRepository downloaded successfully.")

    else:

        print("\nRepository already exists.")

    # -----------------------------------------------------
    # 2. INSPECT REPOSITORY
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("CHECKING REPOSITORY STRUCTURE")
    print("=" * 70)

    print("\nRepository location:")
    print(REPO_DIR)

    print("\nTop-level folders/files:")

    for item in sorted(REPO_DIR.iterdir()):

        if item.name == ".git":
            continue

        if item.is_dir():
            print(f"[DIR ] {item.name}")

        else:
            print(f"[FILE] {item.name}")

    # -----------------------------------------------------
    # 3. SEARCH FOR IMAGE DIRECTORIES
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("SEARCHING FOR IMAGE DATA")
    print("=" * 70)

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    image_files = []

    for file in REPO_DIR.rglob("*"):

        if (
            file.is_file()
            and file.suffix.lower() in image_extensions
        ):
            image_files.append(file)

    print(f"\nImages currently found: {len(image_files)}")

    if not image_files:

        print("\nNo image files were found.")

        print(
            "\nThe repository was cloned, but the actual image dataset "
            "is not included in the Git repository."
        )

        print(
            "\nWe will need to download the images using another method."
        )

        return

    # -----------------------------------------------------
    # 4. SHOW SAMPLE PATHS
    # -----------------------------------------------------

    print("\nFirst 20 image paths:")

    for image in image_files[:20]:

        print(
            image.relative_to(REPO_DIR)
        )

    print("\n" + "=" * 70)
    print("REPOSITORY CHECK COMPLETED")
    print("=" * 70)


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    main()