"""Render only the existing Kaplan–Meier cell, without classifier execution."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import warnings

# Use bundled fonts and an environment-local cache for reproducible rendering.
os.environ["MPL_IGNORE_SYSTEM_FONTS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix) / ".cache/matplotlib"))

import lifelines  # Fail clearly if the locked visualization group is not installed.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import pandas as pd
import seaborn as sns

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "notebooks/exploration.ipynb"
DATA = PROJECT / "data/heart_failure.csv"
CELL_ID = "ffca1914"
COLUMNS = ["time", "DEATH_EVENT", "ejection_fraction"]


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject missing/invalid observations; never impute or silently drop rows."""
    if frame.empty:
        raise ValueError("No observations to plot")
    data = frame.loc[:, COLUMNS].apply(pd.to_numeric, errors="raise")
    if data.isna().any().any() or not data.map(math.isfinite).all().all():
        raise ValueError("Plot columns must be finite and non-missing")
    if not data["time"].gt(0).all():
        raise ValueError("Follow-up duration must be positive, in days")
    if not data["DEATH_EVENT"].isin([0, 1]).all():
        raise ValueError("DEATH_EVENT must be 1 (death) or 0 (right-censored)")
    if not (data["ejection_fraction"].gt(0) & data["ejection_fraction"].le(100)).all():
        raise ValueError("Ejection fraction must be in (0, 100] percent")
    median = data["ejection_fraction"].median()
    if not data["ejection_fraction"].gt(median).any():
        raise ValueError("The median split must have observations in both groups")
    return data


def target_cell(notebook: dict) -> dict:
    cells = [cell for cell in notebook["cells"] if cell.get("id") == CELL_ID]
    if len(cells) != 1 or cells[0]["cell_type"] != "code":
        raise ValueError("Expected exactly one Kaplan–Meier code cell")
    return cells[0]


def render(cell: dict, data: pd.DataFrame) -> bytes:
    """Execute an isolated copy of the one plot cell with its minimal inputs."""
    isolated_cell = copy.deepcopy(cell)
    namespace = {
        "X": data[["time", "ejection_fraction"]].copy(),
        "y": data["DEATH_EVENT"].astype(int).copy(),
        "plt": plt,
    }
    # Match the original notebook's plotting setup without executing its setup,
    # EDA, classifier, evaluation or interpretation cells.
    sns.set_theme(context="notebook", style="whitegrid", font="DejaVu Sans",
                  palette=["#315C82", "#B46A30"])
    plt.close("all")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive, and thus cannot be shown", category=UserWarning)
        exec(compile("".join(isolated_cell["source"]), f"KM cell {CELL_ID}", "exec"), namespace)
    fig = namespace["fig"]
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def with_plot_output(notebook: dict, png: bytes) -> dict:
    """Replace only this cell's output and discard its obsolete execution stamp."""
    updated = copy.deepcopy(notebook)
    cell = target_cell(updated)
    cell["outputs"] = [{
        "output_type": "display_data",
        "data": {
            "image/png": base64.b64encode(png).decode("ascii"),
            "text/plain": ["Kaplan–Meier survival by baseline ejection fraction"],
        },
        "metadata": {},
    }]
    # This is a standalone Python process, not a new full-notebook kernel run.
    cell["execution_count"] = None
    cell["metadata"].pop("execution", None)
    nbformat.validate(nbformat.from_dict(updated))
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Standalone PNG path")
    parser.add_argument("--update-notebook", action="store_true",
                        help="Also replace only the Kaplan–Meier cell output")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        parser.error("--output must be a PNG path")
    original = NOTEBOOK.read_text()
    notebook = json.loads(original)
    nbformat.validate(nbformat.from_dict(notebook))
    data = validate_data(pd.read_csv(DATA, usecols=COLUMNS))
    png = render(target_cell(notebook), data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(png)
    if args.update_notebook:
        updated = with_plot_output(notebook, png)
        if NOTEBOOK.read_text() != original:
            raise RuntimeError("Notebook changed during rendering; refusing to overwrite")
        NOTEBOOK.write_text(json.dumps(updated, indent=1, ensure_ascii=False) + "\n")
    median = float(data["ejection_fraction"].median())
    groups = {}
    for label, mask in [("at_or_below_median", data.ejection_fraction.le(median)),
                        ("above_median", data.ejection_fraction.gt(median))]:
        group = data.loc[mask]
        groups[label] = {"n": len(group), "deaths": int(group.DEATH_EVENT.sum()),
                         "right_censored": int(group.DEATH_EVENT.eq(0).sum())}
    print(json.dumps({
        "python": platform.python_version(), "executable": sys.executable,
        "lifelines": lifelines.__version__, "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "rows": len(data), "follow_up_days": [int(data.time.min()), int(data.time.max())],
        "median_ejection_fraction_percent": median, "groups": groups,
        "output": str(args.output), "updated_notebook": args.update_notebook,
    }, indent=2))


if __name__ == "__main__":
    main()
