# Machine Learning Portfolio

A portfolio of machine-learning projects originally completed for
**CSCI 450 (Machine Learning)** at Lake Forest College, then audited,
refactored, and significantly extended after the course. The work
spans clinical prediction, particle-physics image classification,
classical computer vision, and deep learning.

## Project directory

| Project | Techniques | Dataset | Best result |
|---|---|---|---|
| [`heart_failure_prediction`](projects/heart_failure_prediction/) | Logistic Regression, Random Forest, Gradient Boosting, XGBoost, RBF-SVM. Stratified 5-fold CV with SHAP and Kaplan–Meier analysis. | UCI / Kaggle Heart Failure Clinical Records (299 rows × 13 cols, in-repo) | **Random Forest, ROC-AUC = 0.907 ± 0.017** |
| [`lhc_collision_classification`](projects/lhc_collision_classification/) | HOG features + RF / SVM / XGBoost (Project 1) and ResNet50V2 / MobileNetV2 / custom CNN (Project 2). | Kaggle "Proton Collision 13 TeV" simulated detector images, 3 classes (~30 k images, external download) | **MobileNetV2 (transfer learning), ~0.74 test accuracy** in the original Project 2 |

## Assignments

| Notebook | Topic |
|---|---|
| [`hw1_image_filters_and_retrieval.ipynb`](assignments/hw1_image_filters_and_retrieval.ipynb) | Roberts edge detection, image rescaling, content-based retrieval |
| [`hw2_smoothing_and_tiny_image_retrieval.ipynb`](assignments/hw2_smoothing_and_tiny_image_retrieval.ipynb) | Difference-of-Gaussians, median filter, tiny-image retrieval |
| [`hw3_image_box_drawing.ipynb`](assignments/hw3_image_box_drawing.ipynb) | Random box generation + manual rectangle drawing in NumPy |
| [`hw4_hog_features_and_svm_classifier.ipynb`](assignments/hw4_hog_features_and_svm_classifier.ipynb) | Caltech 101 statistics + HOG / one-vs-rest SVM airplane classifier |
| [`midterm_green_rectangle_detection.ipynb`](assignments/midterm_green_rectangle_detection.ipynb) | Bounding-box detection of a coloured region |

Each assignment notebook now opens with a topic / setup header and
closes with a *“What I'd do differently now”* reflection cell. The
class submissions themselves are otherwise untouched.

## Repository layout

```
MachineLearning/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── assignments/                       ← cleaned coursework notebooks (HW1-4 + midterm)
│
├── projects/
│   ├── heart_failure_prediction/      ← clinical ML, end-to-end
│   │   ├── README.md
│   │   ├── data/heart_failure.csv
│   │   ├── notebooks/
│   │   │   ├── exploration.ipynb
│   │   │   └── 00_original_coursework.ipynb
│   │   ├── results/cv_metrics.csv
│   │   └── src/{data_loader,preprocessing,models,evaluation,train}.py
│   │
│   └── lhc_collision_classification/  ← particle-physics image classification
│       ├── README.md
│       ├── notebooks/
│       │   ├── exploration.ipynb
│       │   ├── 01_classical_ml_baseline.ipynb
│       │   └── 02_deep_learning_models.ipynb
│       └── src/{data_loader,preprocessing,models,evaluation,train}.py
│
├── utils/                             ← shared metric and plotting helpers
│   ├── metrics.py
│   └── visualization.py
│
└── archive/                           ← Deepnote-specific init notebook (not portable)
```

## Setup

```bash
git clone https://github.com/it-malek/MachineLearning.git
cd MachineLearning
pip install -r requirements.txt
```

The heart-failure data ships with the repo. The LHC and assignment
image datasets are too large to bundle; per-project READMEs link to
their download sources.

## Running each project

```bash
# Heart failure: CV benchmark + metrics CSV
python -m projects.heart_failure_prediction.src.train

# Heart failure: full notebook with EDA + SHAP + Kaplan-Meier
jupyter notebook projects/heart_failure_prediction/notebooks/exploration.ipynb

# LHC collisions: pipeline walk-through on a synthetic mini-dataset (no external data needed)
jupyter notebook projects/lhc_collision_classification/notebooks/exploration.ipynb

# LHC collisions: real-data CLI (requires the Kaggle dataset)
python -m projects.lhc_collision_classification.src.train \
    --train-dir ~/data/proton/Train \
    --test-dir  ~/data/proton/Test \
    --image-size 256 \
    --with-pca
```

## What changed vs. the original course submissions

The original notebooks were monolithic class deliverables: a single
`.ipynb` per assignment, no shared utilities, single 80/20 splits, no
hyperparameter tuning, and no AUC reporting. The portfolio version:

* Splits the two main projects into proper Python packages
  (`data_loader`, `preprocessing`, `models`, `evaluation`, `train`).
* Wraps every model in an sklearn `Pipeline` so the scaler is re-fit
  per CV fold — fixing the data-leakage pattern that was implicit
  in the original code.
* Replaces single 70/30 splits with **stratified 5-fold cross-
  validation** with mean ± std reported.
* Adds AUC, calibration curves, threshold sweeps, SHAP values, and
  feature importance plots that the originals didn't have.
* Adds XGBoost and Gradient Boosting comparisons everywhere.
* Provides a CLI entry point per project so results can be
  reproduced without opening Jupyter.
* Includes per-project READMEs with the metric tables and a root
  README (this file) tying the work together.

The original course notebooks are preserved unmodified inside their
project folders (`00_original_coursework.ipynb`,
`01_classical_ml_baseline.ipynb`, `02_deep_learning_models.ipynb`)
so the rebuilt versions can be compared side-by-side.

## Stack

Python 3.10+ • NumPy • pandas • scikit-learn • XGBoost •
SHAP • lifelines • imbalanced-learn • scikit-image • matplotlib •
seaborn • TensorFlow / Keras (LHC deep learning notebook only)

## License

MIT — see [LICENSE](LICENSE).
