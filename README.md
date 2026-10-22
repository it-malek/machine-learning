# Machine-learning coursework

Coursework from Lake Forest College's CSCI 250 and CSCI 450, with later
analysis and reusable utilities. The heart-failure example is retrospective
classification. The collider study lives in
[collider-ml](https://github.com/it-malek/collider-ml).

| Project | Scope |
|---|---|
| [Heart-failure classification](projects/heart_failure_prediction/) | CSCI 250 pair project by Malek Elaghel and Raneem, followed by a cross-validated classifier comparison and a descriptive Kaplan-Meier plot. Saved Random Forest ROC-AUC: **0.907 ± 0.017**. The classifier includes observed follow-up duration, so this is not evidence for prediction at intake. |
| [Assignments](assignments/) | Historical CSCI 450 image-processing notebooks. External images are required for some exercises. |
| [Collider study](https://github.com/it-malek/collider-ml) | Historical image classification and a separate physics-native jet study. See the dedicated repository for its results and limitations. |

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.11.21 or
later. Run commands from the repository root. `pyproject.toml` defines the
dependency groups and `uv.lock` pins their resolution; there is no separate
requirements file or nested project.

| Group | Use | Python |
|---|---|---|
| `dev` (included by default) | Notebook schema and saved-artifact checks | 3.11–3.13 |
| `visualization` | Descriptive Kaplan-Meier plot and its five synthetic checks | 3.11–3.13 |
| `classifiers` | Heart-failure classifiers, shared plotting utilities and SHAP | 3.11–3.13 |
| `notebooks` | Jupyter, IPython kernel and notebook export tools | 3.11–3.13 |
| `images` | NumPy/SciPy, scikit-image and the assignment SVM | 3.11–3.13 |

Choose the groups you need and include them in both `uv sync` and `uv run`.
The environment lives in `.venv`. The examples do not all require the plotting
Python version; select another supported interpreter with `--python`.
On macOS, the XGBoost wheel also requires the OpenMP runtime (`libomp`),
available through `brew install libomp`. This system prerequisite is separate
from uv's Python dependencies and is not needed for the plot-only group.

To reproduce just the saved Kaplan-Meier plot:

```sh
uv sync --locked --python 3.13.14 --group visualization
uv run --locked --group visualization \
  python projects/heart_failure_prediction/scripts/render_kaplan_meier.py \
  --output /tmp/heart-failure-kaplan-meier.png
```

This command runs one plotting cell with its three input columns. It does not
run classifier training, evaluation or the rest of the notebook. The
[heart-failure README](projects/heart_failure_prediction/README.md) explains the
censoring convention and how to update the embedded figure.

To open the heart-failure notebook in the selected environment:

```sh
uv sync --locked --python 3.13 --group classifiers --group visualization --group notebooks
uv run --locked --group classifiers --group visualization --group notebooks \
  jupyter notebook projects/heart_failure_prediction/notebooks/exploration.ipynb
```

Opening the notebook does not execute it. Running its classifier cells recomputes
results; the saved comparison is a historical record, not a claim that current
dependencies reproduce every model metric. The original coursework notebook is
kept separately for reference.

For image coursework, use `--group images --group notebooks` instead. Follow
[the assignment guide](assignments/README.md) for external data and launch commands.

## Checks

```sh
uv sync --locked --python 3.13 --group visualization
uv run --locked python -m unittest discover -s tests -v
uv run --locked --group visualization python -m unittest discover \
  -s projects/heart_failure_prediction/scripts -p 'test_kaplan_meier.py' -v
```

CI also installs the selected groups and checks their imports across supported
Python versions. These checks cover software compatibility, notebook structure,
saved-artifact integrity and the isolated plotting path. The original coursework notebook retains
its Deepnote fields on disk. Only a copy with the identified extension fields
excluded passes the standard schema check; the untouched original is also
checked byte for byte. These checks do not train models
or validate clinical use.

## Attribution

This repository is published as a single source snapshot. Earlier development
history is retained privately.

The heart-failure project began as **CSCI 250 Project 2**, a pair project by
**Malek Elaghel and Raneem**. The original is preserved in
[`00_original_coursework.ipynb`](projects/heart_failure_prediction/notebooks/00_original_coursework.ipynb).
The package, cross-validation and interpretation work were added later.

The image assignments and original collider notebooks began as individual
**CSCI 450** submissions in Fall 2023. Collider artifacts and their chronology
are documented in the dedicated repository.

Dataset attribution is separate from the [MIT code license](LICENSE):
Chicco & Jurman (2020) via UCI/Kaggle for heart-failure records, Oxford 102 Flowers
and Caltech 101 for the image exercises, and the sources documented in collider-ml
for the collider studies.
