# Heart-failure retrospective classification

This project began as **CSCI 250 Project 2**, a pair project by **Malek Elaghel
and Raneem**. The original notebook is preserved in
[`00_original_coursework.ipynb`](notebooks/00_original_coursework.ipynb). The
Python package and cross-validated comparison were added after the course.

The later analysis classifies recorded mortality (`DEATH_EVENT`) using the
[Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
dataset from Chicco & Jurman (2020). It uses 12 columns, including observed
follow-up duration (`time`), which is unavailable at hospital intake.

## Dataset

| | |
|---|---|
| Source | UCI / Kaggle; Chicco & Jurman (2020) |
| Samples | 299 |
| Features | 7 continuous + 5 binary |
| Target | `DEATH_EVENT` (binary, 32% positive) |
| Missing values | 0 |
| Stored at | `data/heart_failure.csv` |

## Saved classifier comparison

The pipeline fits scaling within each training fold and compares five
classifiers using stratified 5-fold cross-validation. This prevents scaler leakage across folds; it does not
make follow-up duration available at intake.

**The saved results below describe retrospective classification including observed
follow-up duration. They do not establish baseline intake mortality prediction.**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. Precision |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.796 ± 0.037 | 0.669 ± 0.076 | 0.771 ± 0.055 | 0.710 ± 0.024 | 0.878 ± 0.033 | 0.788 ± 0.066 |
| **Random Forest** | **0.849 ± 0.039** | **0.807 ± 0.064** | 0.697 ± 0.098 | **0.745 ± 0.075** | **0.907 ± 0.017** | **0.833 ± 0.042** |
| Gradient Boosting | 0.829 ± 0.033 | 0.760 ± 0.053 | 0.687 ± 0.100 | 0.718 ± 0.067 | 0.876 ± 0.031 | 0.768 ± 0.051 |
| XGBoost | 0.829 ± 0.033 | 0.757 ± 0.047 | 0.687 ± 0.094 | 0.718 ± 0.067 | 0.901 ± 0.022 | 0.809 ± 0.028 |
| SVM (RBF) | 0.789 ± 0.027 | 0.642 ± 0.040 | 0.792 ± 0.073 | 0.707 ± 0.035 | 0.876 ± 0.043 | 0.756 ± 0.094 |

The original coursework reported a single 70/30 split, no AUC and about 80%
accuracy on selected feature subsets. The later pipeline's saved ROC-AUC of
approximately 0.91 includes follow-up duration. Feature importance, calibration
and threshold plots describe that fitted classifier; they do not establish
clinical actionability or a screening threshold.

The confusion matrix and threshold-sweep figures in notebook section 6 are
presentation derivatives with revised titles and unchanged plotted regions.

## Descriptive Kaplan-Meier plot

The plot in [section 8 of the exploration notebook](notebooks/exploration.ipynb)
uses follow-up days as the time variable. `DEATH_EVENT=1` denotes observed death;
`0` is treated as right-censored at the recorded follow-up time. This is an
appropriate use of observed duration for survival analysis, distinct from using
it as an intake predictor.

The existing sample-median split is EF 38%: 182 records have EF ≤ 38% and 117 have
EF > 38%. The 299 records contain 96 observed deaths and 203 censored observations,
with follow-up from 4 to 285 days. All three plot columns pass the input checks;
no rows are imputed or dropped. Shading shows lifelines'
[95% log-log confidence intervals](https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html).

The comparison is unadjusted and descriptive, and the sample median is not a
clinical cutoff. Kaplan-Meier interpretation assumes non-informative censoring
within each group. The records do not verify that assumption or identify
individual censoring reasons; the
[dataset paper](https://doi.org/10.1186/s12911-020-1023-5) notes limited
follow-up-protocol information.

## Reproduce only the plot

From the repository root, using the authoritative root `pyproject.toml` and
`uv.lock`:

```sh
uv sync --locked --python 3.13.14 --group visualization
uv run --locked --group visualization \
  python projects/heart_failure_prediction/scripts/render_kaplan_meier.py \
  --output /tmp/heart-failure-kaplan-meier.png
```

The runner selects only the Kaplan-Meier cell and prepares its three input
columns: `time`, `DEATH_EVENT` and `ejection_fraction`. It uses the uv environment,
a non-interactive backend and bundled fonts. It does not run EDA, classifier,
evaluation or SHAP cells. The saved figure was rendered with Python 3.13.14 and
lifelines 0.30.3; the visualization group also supports Python 3.11 and 3.12.

Add `--update-notebook` only when you intend to refresh the embedded plot. That
option replaces the plot cell's output and clears its old execution stamp.
All other outputs, execution counts and notebook metadata stay unchanged.

Run the five synthetic plotting checks with:

```sh
uv run --locked --group visualization python -m unittest discover \
  -s projects/heart_failure_prediction/scripts -p 'test_kaplan_meier.py' -v
```

## Classifier and notebook setup

Use the `classifiers`, `visualization` and `notebooks` groups to open the complete
exploration notebook. See the [root setup guide](../../README.md#setup). Installing
the environment does not revalidate the historical model results.

The CLI below deliberately runs classifier cross-validation and overwrites
`results/cv_metrics.csv`; it is separate from plot reproduction:

```sh
uv run --locked --group classifiers python -m projects.heart_failure_prediction.src.train
```

## Files

- `src/`: dataset loader, preprocessing, model definitions, evaluation and CLI.
- `notebooks/exploration.ipynb`: later analysis with saved results and figures.
- `notebooks/00_original_coursework.ipynb`: original pair coursework.
- `scripts/`: isolated plot runner and synthetic checks.
- `data/heart_failure.csv`: dataset supplied with the example.
- `results/cv_metrics.csv`: saved classifier comparison.
