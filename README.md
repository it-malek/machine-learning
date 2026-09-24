# Machine Learning Portfolio

A portfolio of machine-learning projects that began as Lake Forest
College coursework — **CSCI 250 (Programming for Data Applications)**
and **CSCI 450 (Computer Vision & Machine Learning)** — and were
reorganized, refactored, and extended after the courses ended. The
work spans clinical prediction, particle-physics image classification,
classical computer vision, and deep learning. This repository remains the umbrella
for coursework and smaller projects. The collider project now lives in its dedicated
repository, [collider-ml](https://github.com/it-malek/collider-ml), with its historical
notebooks, corrected record and completed modern extension. Earlier versions remain
in this repository's Git history.

## Project directory

| Project | Techniques | Dataset | Best result |
|---|---|---|---|
| [`heart_failure_prediction`](projects/heart_failure_prediction/) | Logistic Regression, Random Forest, Gradient Boosting, XGBoost, RBF-SVM. Stratified 5-fold CV with SHAP; optional Kaplan–Meier survival curve (requires `lifelines`). | UCI / Kaggle Heart Failure Clinical Records (299 rows × 13 cols, in-repo) | **Random Forest, ROC-AUC = 0.907 ± 0.017** |
| [collider-ml](https://github.com/it-malek/collider-ml) — dedicated repository | Historical image classification, reproducibility auditing and physics-native jet classification. | Historical screenshots and ATLAS simulated jets; different classification tasks. | See the dedicated repository for the corrected results and unweighted-only Test limitations. |

## Assignments

| Notebook | Topic |
|---|---|
| [`hw1_image_filters_and_retrieval.ipynb`](assignments/hw1_image_filters_and_retrieval.ipynb) | Roberts edge detection, image rescaling, content-based retrieval |
| [`hw2_smoothing_and_tiny_image_retrieval.ipynb`](assignments/hw2_smoothing_and_tiny_image_retrieval.ipynb) | Difference-of-Gaussians, median filter, tiny-image retrieval |
| [`hw3_image_box_drawing.ipynb`](assignments/hw3_image_box_drawing.ipynb) | Random box generation + manual rectangle drawing in NumPy |
| [`hw4_hog_features_and_svm_classifier.ipynb`](assignments/hw4_hog_features_and_svm_classifier.ipynb) | Caltech 101 statistics + HOG / one-vs-rest SVM airplane classifier |
| [`midterm_green_rectangle_detection.ipynb`](assignments/midterm_green_rectangle_detection.ipynb) | Bounding-box detection of a coloured region |

Each assignment notebook opens with a short header describing the
topic and how to obtain its dataset. The class submissions themselves
are otherwise untouched.

## Repository layout

```
machine-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── assignments/                       ← cleaned coursework notebooks (HW1-4 + midterm)
│
├── projects/
│   └── heart_failure_prediction/      ← clinical ML, end-to-end
│       ├── README.md
│       ├── data/heart_failure.csv
│       ├── notebooks/
│       │   ├── exploration.ipynb
│       │   └── 00_original_coursework.ipynb
│       ├── results/cv_metrics.csv
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
git clone https://github.com/it-malek/machine-learning.git
cd machine-learning
pip install -r requirements.txt
```

The heart-failure data ships with the repo. Assignment image datasets are
external; the assignment guide links to their sources. Collider data provenance
and reproducibility boundaries are documented in the dedicated repository.

## Running each project

```bash
# Heart failure: CV benchmark + metrics CSV
python -m projects.heart_failure_prediction.src.train

# Heart failure: notebook with EDA + SHAP (Kaplan-Meier cell runs only if lifelines is installed)
jupyter notebook projects/heart_failure_prediction/notebooks/exploration.ipynb

```

## Origins and attribution

* `projects/heart_failure_prediction/` began as **CSCI 250 Project 2, a
  pair project by Malek Elaghel and Raneem**. The original notebook is
  preserved unmodified as `notebooks/00_original_coursework.ipynb`. The
  `src/` package, the cross-validated benchmark, and the SHAP analysis
  were added after the course.
* `assignments/` and the original collider notebooks began as my individual
  CSCI 450 submissions (Fall 2023). The collider artifacts and their chronology
  are now documented in [collider-ml](https://github.com/it-malek/collider-ml).
* Datasets belong to their cited sources (Chicco & Jurman 2020 via
  UCI/Kaggle; Oxford 102 Flowers; Caltech 101). Collider dataset attribution is
  documented separately in the dedicated repository.

## What changed vs. the original course submissions

The original notebooks were monolithic class deliverables: a single
`.ipynb` per assignment, no shared utilities, single 80/20 splits, no
hyperparameter tuning, and no AUC reporting. The portfolio version:

* Organizes the heart-failure project into a Python package
  (`data_loader`, `preprocessing`, `models`, `evaluation`, `train`).
* Wraps every model in an sklearn `Pipeline` so the scaler is re-fit
  per CV fold — fixing the data-leakage pattern that was implicit
  in the original code.
* Replaces single 70/30 splits with **stratified 5-fold cross-
  validation** with mean ± std reported.
* Adds AUC, calibration curves, threshold sweeps, SHAP values, and
  feature importance plots that the originals didn't have.
* Adds XGBoost and Gradient Boosting comparisons everywhere.
* Provides a CLI entry point for the heart-failure project so results can be
  reproduced without opening Jupyter.
* Includes per-project READMEs with the metric tables and a root
  README (this file) tying the work together.

The original heart-failure notebook remains in its project folder as
`00_original_coursework.ipynb`. Original collider notebooks are preserved in
the dedicated repository and in this repository's prior history.

## Stack

Python 3.10+ • NumPy • pandas • scikit-learn • XGBoost •
SHAP • scikit-image • matplotlib • seaborn • lifelines (optional)

## License

MIT — see [LICENSE](LICENSE).
