"""Check public artifact integrity without executing notebooks or classifiers."""

import ast
import hashlib
import json
from pathlib import Path
import re
import unittest

import nbformat

ROOT = Path(__file__).resolve().parents[1]
# Freeze the presentation editions, original pair coursework, data and metrics.
FIXED_ARTIFACTS = {
    "assignments/hw1_image_filters_and_retrieval.ipynb": "c9ac192a3e54db53045fc8f11b906559a0277b2f37024f5726a94311b76275dc",
    "assignments/hw2_smoothing_and_tiny_image_retrieval.ipynb": "01102e3b17f1cd3a1cd4a1f9f57bb3be056e32426c71d667a036c62071d5fa6d",
    "assignments/hw3_image_box_drawing.ipynb": "d9efffb56a31d987491373ce88c8793b0e675c714c57deda61f2b52904251206",
    "assignments/hw4_hog_features_and_svm_classifier.ipynb": "9429de6fc87271c5ef020c077401aba8df5a3531e388930748032e907798a03e",
    "assignments/midterm_green_rectangle_detection.ipynb": "7fee36a18c61963624929564f142955490eb9e2a865bf7507da8191422be4781",
    "projects/heart_failure_prediction/data/heart_failure.csv": "9c73cea7468ff5d517801ec050fe9993da5912fce4b56f296f8df3b38dd75912",
    "projects/heart_failure_prediction/notebooks/00_original_coursework.ipynb": "a507f5e737599fe95d16b6c7648ca9a6612b5bad1e8639cc1734a3ac458964a1",
    "projects/heart_failure_prediction/results/cv_metrics.csv": "aea2a7295e725164014452e2498e31b1d5cd60ef1089d16c2c27edec464d0941"
}
EXPLORATION_RECORD = "48264fd290597c82685f85dcb5612eaba1a314ae2d4ec096dfab53f097d0a9a2"


class PublicArtifactTests(unittest.TestCase):
    def test_publication_notebooks_data_and_metrics_are_preserved(self):
        for name, expected in FIXED_ARTIFACTS.items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)

    def test_saved_figure_editions_outputs_and_metadata_are_preserved(self):
        notebook = json.loads((ROOT / "projects/heart_failure_prediction/notebooks/exploration.ipynb").read_text())
        record = [{key: cell[key] for key in ("outputs", "execution_count", "metadata") if key in cell}
                  for cell in notebook["cells"]]
        self.assertEqual(hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(), EXPLORATION_RECORD)

    def test_notebook_schemas(self):
        paths = [*ROOT.glob("assignments/*.ipynb"),
                 *ROOT.glob("projects/heart_failure_prediction/notebooks/*.ipynb")]
        for path in paths:
            with self.subTest(path=path.name):
                notebook = nbformat.read(path, as_version=4)
                if path.name == "00_original_coursework.ipynb":
                    # Preserve the original on disk. Validate standard fields
                    # on an in-memory copy without their known extension properties.
                    for cell in notebook.cells:
                        cell.pop("block_group", None)
                        cell.pop("outputs_reference", None)
                nbformat.validate(notebook)

    def test_maintained_python_syntax(self):
        for directory in ("utils", "scripts", "tests", "projects/heart_failure_prediction/src",
                          "projects/heart_failure_prediction/scripts"):
            for path in (ROOT / directory).glob("*.py"):
                with self.subTest(path=path.name):
                    ast.parse(path.read_text(), filename=str(path))

    def test_document_links(self):
        for name in ("README.md", "assignments/README.md", "projects/heart_failure_prediction/README.md"):
            path = ROOT / name
            for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                if "://" not in link and not link.startswith("#"):
                    with self.subTest(document=name, link=link):
                        self.assertTrue((path.parent / link.split("#")[0]).exists())


if __name__ == "__main__":
    unittest.main()
