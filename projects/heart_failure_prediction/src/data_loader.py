"""Heart failure dataset loader.

The CSV ships with the project under ``projects/heart_failure_prediction/data/``,
so no download is required. The loader also validates the schema and
emits a small descriptive summary so callers can sanity-check that the
file hasn't been replaced or corrupted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# Default location relative to this file.
DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "heart_failure.csv"
)

TARGET = "DEATH_EVENT"

BINARY_FEATURES = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking",
]

CONTINUOUS_FEATURES = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time",
]

FEATURE_COLUMNS = CONTINUOUS_FEATURES + BINARY_FEATURES


@dataclass
class DatasetSummary:
    """Lightweight description of the loaded dataset."""

    n_rows: int
    n_features: int
    class_balance: dict[int, int]
    n_missing: int

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DatasetSummary(rows={self.n_rows}, features={self.n_features}, "
            f"class_balance={self.class_balance}, missing={self.n_missing})"
        )


def load_heart_failure(
    path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, DatasetSummary]:
    """Load the heart failure CSV and split into ``X`` / ``y``.

    Args:
        path: Optional override for the CSV location. If ``None`` we
            resolve the default packaged-with-the-repo path.

    Returns:
        Tuple ``(X, y, summary)`` where ``X`` is a ``DataFrame`` with
        only the modelling features (in a deterministic column order),
        ``y`` is the binary ``DEATH_EVENT`` series, and ``summary`` is a
        :class:`DatasetSummary` for logging / sanity checks.

    Raises:
        FileNotFoundError: If the CSV cannot be located.
        ValueError: If any expected column is missing.
    """
    csv_path = Path(path) if path is not None else DEFAULT_CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Heart failure dataset not found at {csv_path}. "
            "Pass an explicit path or restore the file under data/."
        )

    df = pd.read_csv(csv_path)

    expected = set(FEATURE_COLUMNS + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Heart failure CSV is missing required columns: {sorted(missing)}"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(int).copy()

    summary = DatasetSummary(
        n_rows=len(df),
        n_features=len(FEATURE_COLUMNS),
        class_balance=y.value_counts().to_dict(),
        n_missing=int(df.isna().sum().sum()),
    )
    logger.info("Loaded heart failure data: %s", summary)
    return X, y, summary


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.INFO)
    X, y, summary = load_heart_failure()
    print(summary)
    print(X.head())
