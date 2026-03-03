import os
import shutil


def filter_dataset():
  old_root = "Data"
  new_root = "data0"

  subfolders = ["Aneuploid", "Euploid"]

  for sub in subfolders:
    old_path = os.path.join(old_root, sub)
    new_path = os.path.join(new_root, sub + "0")
    os.makedirs(new_path, exist_ok=True)

    for file in os.listdir(old_path):
      if file.endswith("0.png"):
        src = os.path.join(old_path, file)
        dest = os.path.join(new_path, file)
        shutil.copy2(src, dest)


  print("Filtering complete.")

if __name__ == "__main__":
  filter_dataset()