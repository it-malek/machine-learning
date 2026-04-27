# LHC Collision Image Classification

Classification of simulated 13 TeV proton-proton collision detector
images from CERN Open Data into three event types:

| Class | Physics |
|---|---|
| QCD   | Quantum Chromodynamics multi-jet background |
| TTbar | Top–antitop pair production |
| WJets | W boson + jets |

This project is the rebuilt, modular version of the original CSCI 450
*Project 1* (classical ML — HOG + RF/SVM) and *Project 2* (CNN ensembles).

## Dataset

The full image dataset (~30k 372×1196 grayscale PNGs) is **not bundled
with the repo** because of size. Download it from Kaggle:

> https://www.kaggle.com/datasets/anilkumarpatibandla/proton-collision-image-set

Unpack it so the layout is:

```
~/data/proton/
├── Train/
│   ├── QCD/   image_*.png
│   ├── TTbar/ image_*.png
│   └── WJets/ image_*.png
└── Test/
    └── (same shape)
```

Then either run the CLI (below) or point the notebooks' `train_path`
variable at your `Train/` directory.

## Approach

The rebuilt pipeline mirrors the heart-failure project's structure:

1. **`data_loader.py`** — walk the class folders and return file paths
   + integer labels. Streaming-friendly, no all-images-in-RAM.
2. **`preprocessing.py`** — resize images and extract HOG descriptors
   (8 orientations, 32×32 cells, configurable). Also handles
   grayscale ↔ RGB normalisation.
3. **`models.py`** — sklearn `Pipeline` factories for Random Forest,
   Linear SVM, RBF SVM (with PCA), Gradient Boosting, and XGBoost.
   Every model has a `StandardScaler` head — important for SVMs since
   HOG bin magnitudes vary widely.
4. **`evaluation.py`** — stratified 5-fold cross-validation with
   accuracy, macro-F1, weighted-F1, and balanced accuracy.
5. **`train.py`** — CLI entry point.

## Notebooks

| Notebook | Purpose |
|---|---|
| `exploration.ipynb` | Walk-through of the rebuilt pipeline on a tiny synthetic dataset. **Executes anywhere** — no real data required. |
| `01_classical_ml_baseline.ipynb` | Original course Project 1 (HOG + RF/SVM with/without PCA). Preserved unmodified. |
| `02_deep_learning_models.ipynb` | Original course Project 2 (ResNet50V2, custom CNN, MobileNetV2). Preserved unmodified. |

## Reproducing the results

```bash
pip install -r requirements.txt

# CLI on the real dataset
python -m projects.lhc_collision_classification.src.train \
    --train-dir ~/data/proton/Train \
    --test-dir  ~/data/proton/Test \
    --image-size 256 \
    --with-pca

# Walk-through notebook (no dataset required, runs synthetic data)
jupyter notebook projects/lhc_collision_classification/notebooks/exploration.ipynb
```

## Original course results (for context)

The class submissions reported on the **untuned, single-split** test set:

| Model (original notebook) | Test accuracy |
|---|---|
| Random Forest, HOG only | ~0.46 |
| Random Forest, HOG + PCA(100) | ~0.48 |
| Linear SVM, HOG only | ~0.45 |
| Linear SVM, HOG + PCA(100) | ~0.48 |
| MobileNetV2 (Project 2 winner) | ~0.74 |

The rebuilt pipeline reports the same table cleanly, with stratified
5-fold cross-validation (so each number comes with a standard
deviation), proper class-balanced loss weighting, and a single
`Pipeline` per model so there's no leakage between splits.

## What was changed vs. the original notebooks

* Replaced ad hoc loops with `Pipeline` + `cross_validate`.
* Added `StandardScaler` consistently — the original SVMs ran on
  raw HOG vectors, which is suboptimal.
* Added `class_weight='balanced'` everywhere (the dataset is
  approximately balanced but it's the right default).
* Added XGBoost and Gradient Boosting models for comparison.
* Removed ~12 cells of commented-out exploratory code that the
  original notebooks accumulated during the assignment.

## Files

```
lhc_collision_classification/
├── README.md
├── notebooks/
│   ├── exploration.ipynb              ← portfolio notebook (synthetic-data demo)
│   ├── 01_classical_ml_baseline.ipynb ← original Project 1
│   └── 02_deep_learning_models.ipynb  ← original Project 2
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── models.py
    ├── evaluation.py
    └── train.py
```
