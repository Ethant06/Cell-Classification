<h1 align="center">
	<br>
	Enhancing CNN Performance on cellular classification through Augmentation and Regularization Techniques
	</br>
</h1>

---


## Required Installations
1. torch
2. torchvision
3. matplotlib
4. scikit-learn
4. numpy
5. pyyaml
6. pip install torch torchvision matplotlib scikit-learn numpy pyyaml

## Experiments in configs/:
- baseline (trains with full training split, no data augmentation/regularization)
- small_aug_reg (trains with stratified subset of training split, data augmentation + regularization)
- small_flip (trains with stratified subset of training split, data augmentation)
- small_no_aug (trains with stratified subset of training split, baseline - no data augmentation/regularization)
- small_rotation (trains with stratified subset of training split, data augmentation)
- small_with_aug (trains with stratified subset of training split, data augmentation)


## To Run Experiment
1. Create folder Data/, and within Data/ initiate Euploid/ and Aneuploid/ folders
2. Add Aneuploid images to Data/Aneuploid folder and Euploid images to Data/Euploid folder
3. If not exist already, create folder data2/. Execute scripts/filter_dataset.py to filter all images in time frame _8 to data2/ folder
4. Execute main.py

## Main Files to debug

```md
├── main.py                # Runs all experiment across several seeds
├── src/
│   ├── dataset.py         # Dataset loading, splitting, and augmentation
│   ├── model.py           # CNN architecture
│   ├── train.py           # Training loop + training plots
│   └── evaluate.py        # Testing loop + accuracy + testing plots

```

# Others
```md
├── configs/               # YAML configs for each experiment
├── saved_splits/          # Fixed train/test indices for data2 image extraction

```

## Seed Usage

```md

At the start of each run these are seeded:
1. torch
2. numpy
3. python random

Purpose:
  - Control randomness from:
  - model initialization
  - data shuffling
  - augmentation
  - reduced subset selection


