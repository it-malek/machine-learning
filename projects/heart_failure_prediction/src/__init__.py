"""Heart failure prediction project.

A clinical prediction pipeline trained on the UCI / Kaggle "Heart Failure
Clinical Records" dataset (Chicco & Jurman, 2020). The goal is to
predict in-study mortality (``DEATH_EVENT``) from 12 demographic and
clinical features collected at baseline.

Submodules:
    data_loader:    Load and validate the CSV.
    preprocessing:  Build sklearn ``ColumnTransformer`` pipelines.
    models:         Factory for the model zoo we benchmark.
    evaluation:     Cross-validated evaluation + interpretation.
    train:          End-to-end training entry point (``python -m ...``).
"""
