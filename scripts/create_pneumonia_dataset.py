"""Create the flat PneumoniaMNIST dataset used by the experiments.

The official source splits are downloaded and merged into ``normal`` and
``pneumonia`` class directories. The training pipeline subsequently creates its
own fixed stratified split to reproduce the completed study design.
"""

from pathlib import Path

from medmnist import PneumoniaMNIST

# PneumoniaMNIST's binary labels and the corresponding ImageFolder names.
LABEL_MAP = {
    0: "normal",
    1: "pneumonia",
}

# All official source splits are exported into one flat collection.
SOURCE_SPLITS = ("train", "val", "test")


def create_dataset(output_dir: str | Path = "data4") -> int:
    """Download and export PneumoniaMNIST in a flat ImageFolder layout.

    Args:
        output_dir: Destination root for ``normal`` and ``pneumonia`` folders.

    Returns:
        Number of images written. A single global counter names files across
        source splits (for example, ``normal123.png``).
    """
    output_dir = Path(output_dir)
    for class_name in LABEL_MAP.values():
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    image_index = 0
    for split in SOURCE_SPLITS:
        dataset = PneumoniaMNIST(split=split, download=True, size=128)
        for image, label in dataset:
            class_name = LABEL_MAP[int(label.item())]
            output_path = output_dir / class_name / f"{class_name}{image_index}.png"
            image.save(output_path)
            image_index += 1

    print(f"Created {image_index} images in {output_dir}.")
    return image_index


if __name__ == "__main__":
    create_dataset()
