"""Extract the cell-image timepoint used by the experiments.

The raw source must use ``Data/Aneuploid`` and ``Data/Euploid`` directories.
Images whose filenames end in ``8.png`` are copied into the two-class
``data2`` ImageFolder layout expected by the cell configurations.
"""

import shutil
from pathlib import Path

RAW_ROOT = Path("Data")
OUTPUT_ROOT = Path("data2")
SOURCE_CLASSES = ("Aneuploid", "Euploid")
TIMEPOINT_SUFFIX = "8.png"


def filter_dataset() -> int:
    """Copy the selected timepoint into ``data2`` and return the image count.

    Only top-level files are inspected. Output class names append ``2`` to the
    source names, yielding ``Aneuploid2`` and ``Euploid2``.

    Raises:
        FileNotFoundError: If either required raw class directory is absent.
    """
    copied = 0
    for class_name in SOURCE_CLASSES:
        source_dir = RAW_ROOT / class_name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source directory: {source_dir}")

        destination_dir = OUTPUT_ROOT / f"{class_name}2"
        destination_dir.mkdir(parents=True, exist_ok=True)

        for source_path in source_dir.iterdir():
            if source_path.is_file() and source_path.name.endswith(TIMEPOINT_SUFFIX):
                shutil.copy2(source_path, destination_dir / source_path.name)
                copied += 1

    print(f"Copied {copied} images into {OUTPUT_ROOT}.")
    return copied


if __name__ == "__main__":
    filter_dataset()