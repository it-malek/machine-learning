"""Check dependency groups and local imports without loading datasets or fitting models."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

GROUPS = {
    "classifiers": [
        "numpy", "pandas", "scipy", "matplotlib", "seaborn", "sklearn",
        "xgboost", "shap", "projects.heart_failure_prediction.src.train", "utils.metrics",
    ],
    "images": ["numpy", "scipy", "matplotlib", "skimage", "sklearn"],
    "notebooks": ["jupyter", "ipykernel", "nbconvert", "nbformat"],
    "visualization": ["lifelines", "pandas", "matplotlib", "seaborn", "nbformat"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", nargs="+", choices=GROUPS)
    args = parser.parse_args()
    env = dict(os.environ, MPLBACKEND="Agg", MPL_IGNORE_SYSTEM_FONTS="1")
    env["MPLCONFIGDIR"] = str(Path(sys.prefix) / ".cache/matplotlib")
    code = """import faulthandler, importlib, sys
faulthandler.dump_traceback_later(45, exit=True)
for name in sys.argv[1:]:
    importlib.import_module(name)
    print('Imported', name, flush=True)
faulthandler.cancel_dump_traceback_later()
"""
    for group in args.groups:
        print(f"Checking {group} on Python {sys.version.split()[0]}", flush=True)
        subprocess.run([sys.executable, "-c", code, *GROUPS[group]],
                       env=env, check=True, timeout=60)


if __name__ == "__main__":
    main()
