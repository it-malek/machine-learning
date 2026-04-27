# Heart Failure Prediction

Cross-validated mortality prediction on the [Heart Failure Clinical
Records](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)
dataset (Chicco & Jurman, 2020).

## Problem

Given 12 baseline clinical and demographic features (age, ejection
fraction, serum creatinine, etc.) recorded at hospital intake, predict
whether the patient will die during the follow-up period (`DEATH_EVENT`).

## Dataset

| | |
|---|---|
| Source | UCI / Kaggle — Chicco & Jurman, 2020 |
| Samples | 299 |
| Features | 7 continuous + 5 binary |
| Target | `DEATH_EVENT` (binary, 32% positive) |
| Missing values | 0 |
| Stored at | `data/heart_failure.csv` |

## Approach

1. **Preprocessing** — `StandardScaler` on continuous features, identity
   passthrough on binary features, both wrapped in a
   `ColumnTransformer` so it can be re-fit per fold without leakage
   ([`src/preprocessing.py`](src/preprocessing.py)).
2. **Models benchmarked** — Logistic Regression, Random Forest, Gradient
   Boosting, XGBoost, RBF-kernel SVM
   ([`src/models.py`](src/models.py)).
3. **Evaluation** — Stratified 5-fold cross-validation; AUC, average
   precision, F1, precision, recall, accuracy
   ([`src/evaluation.py`](src/evaluation.py)).
4. **Interpretation** — Feature importances + SHAP summary and waterfall
   plots, plus a Kaplan-Meier survival curve
   ([`notebooks/exploration.ipynb`](notebooks/exploration.ipynb)).

## Results (5-fold cross-validation)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. Precision |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.796 ± 0.037 | 0.669 ± 0.076 | 0.771 ± 0.055 | 0.710 ± 0.024 | 0.878 ± 0.033 | 0.788 ± 0.066 |
| **Random Forest** | **0.849 ± 0.039** | **0.807 ± 0.064** | 0.697 ± 0.098 | **0.745 ± 0.075** | **0.907 ± 0.017** | **0.833 ± 0.042** |
| Gradient Boosting | 0.829 ± 0.033 | 0.760 ± 0.053 | 0.687 ± 0.100 | 0.718 ± 0.067 | 0.876 ± 0.031 | 0.768 ± 0.051 |
| XGBoost | 0.829 ± 0.033 | 0.757 ± 0.047 | 0.687 ± 0.094 | 0.718 ± 0.067 | 0.901 ± 0.022 | 0.809 ± 0.028 |
| SVM (RBF) | 0.789 ± 0.027 | 0.642 ± 0.040 | 0.792 ± 0.073 | 0.707 ± 0.035 | 0.876 ± 0.043 | 0.756 ± 0.094 |

The original coursework notebook reported a single 70/30 split with no
AUC and ~80% accuracy on hand-picked feature subsets; the rebuilt
pipeline recovers a robust **ROC-AUC ≈ 0.91** with proper CV.

## Reproducing the results

From the repo root:

```bash
pip install -r requirements.txt

# CLI: prints the metrics table and writes results/cv_metrics.csv
python -m projects.heart_failure_prediction.src.train

# Notebook: full EDA + plots + SHAP + Kaplan-Meier
jupyter notebook projects/heart_failure_prediction/notebooks/exploration.ipynb
```

## Key findings

* **Random Forest** wins on every metric except recall (where SVM is a
  hair higher). Gains over the linear baseline are modest, suggesting
  the dataset is mostly linearly separable in scaled feature space.
* **Top drivers** of predicted mortality (by SHAP magnitude) are
  `time`, `serum_creatinine`, and `ejection_fraction`. The latter two
  are clinically actionable; `time` is a study-design proxy and
  including it in deployment would leak survival information.
* A **decision threshold below 0.5** trades precision for recall,
  preferable in a screening setting where missing an at-risk patient
  is more costly than a false alarm. The threshold sweep in the
  notebook makes this explicit.

## Files

```
heart_failure_prediction/
├── README.md
├── data/heart_failure.csv
├── notebooks/
│   ├── exploration.ipynb           ← portfolio notebook (EDA + benchmark + SHAP)
│   └── 00_original_coursework.ipynb ← unmodified original for reference
├── results/cv_metrics.csv          ← latest CV summary written by train.py
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── models.py
    ├── evaluation.py
    └── train.py
```
