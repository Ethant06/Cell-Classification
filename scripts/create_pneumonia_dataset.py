"""
Create pneumonia dataset from PneumoniaMNIST, preserving the official train/val/test splits.
Using a single merged folder and then re-splitting 70/30 would mix train/val/test and can cause
patient/sample leakage (same patient in train and test), inflating accuracy.
We write to data4/train/, data4/val/, data4/test/ so the pipeline can use the official split.
"""
import os
from PIL import Image
from medmnist import PneumoniaMNIST

label_map = {
    0: "normal",
    1: "pneumonia",
}

splits = ["train", "val", "test"]

for split in splits:
    for cls in label_map.values():
        os.makedirs(os.path.join("data4", split, cls), exist_ok=True)

idx = 0
for split in splits:
    dataset = PneumoniaMNIST(split=split, download=True, size=128)
    for img, label in dataset:
        label = int(label.item())
        class_name = label_map[label]
        out_path = os.path.join("data4", split, class_name, f"{class_name}_{idx}.png")
        img.save(out_path)
        idx += 1
