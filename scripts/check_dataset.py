from torchvision.datasets import ImageFolder
import os

ds = ImageFolder("data2")
print("Classes:", ds.classes)
print("Number of images:", len(ds))