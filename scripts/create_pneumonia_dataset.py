import os
from PIL import Image
from medmnist import PneumoniaMNIST

label_map = {
  0: "normal",
  1: "pneumonia"
}

os.makedirs(os.path.join('data4', 'normal'), exist_ok= True)
os.makedirs(os.path.join('data4', 'pneumonia'), exist_ok = True)

splits = ["train", "val", "test"]

idx = 0
for split in splits:
    dataset = PneumoniaMNIST(split=split, download=True, size=128)

    for img, label in dataset:
        label = int(label.item())
        class_name = label_map[label]
        img.save(os.path.join('data4', class_name, f"{class_name}{idx}.png"))
        idx += 1