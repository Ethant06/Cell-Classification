"""Create the flat PneumoniaMNIST dataset used by the experiment configs."""

import os

from medmnist import PneumoniaMNIST

label_map = {
    0: "normal",
    1: "pneumonia",
}

splits = ["train", "val", "test"]


def create_dataset(output_dir="data4"):
    """Merge the source splits into the two-class ImageFolder layout used by this project."""
    for class_name in label_map.values():
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

    image_index = 0
    for split in splits:
        dataset = PneumoniaMNIST(split=split, download=True, size=128)
        for image, label in dataset:
            class_name = label_map[int(label.item())]
            output_path = os.path.join(
                output_dir, class_name, f"{class_name}{image_index}.png"
            )
            image.save(output_path)
            image_index += 1

    print(f"Created {image_index} images in {output_dir}.")


if __name__ == "__main__":
    create_dataset()
