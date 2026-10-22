"""Heart failure retrospective classification project.

A retrospective classification pipeline for the UCI / Kaggle "Heart Failure
Clinical Records" dataset (Chicco & Jurman, 2020). It classifies recorded
in-study mortality (``DEATH_EVENT``) using 12 features, including observed
follow-up duration (``time``). This duration is unavailable at intake, so
the saved results do not establish baseline intake mortality prediction.

Submodules:
    data_loader:    Load and validate the CSV.
    preprocessing:  Build sklearn ``ColumnTransformer`` pipelines.
    models:         Factory for the model zoo we benchmark.
    evaluation:     Cross-validated evaluation + interpretation.
    train:          End-to-end training entry point (``python -m ...``).
"""
