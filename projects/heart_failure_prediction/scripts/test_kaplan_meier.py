"""Synthetic checks for plot input handling and preservation of other cells."""
import copy
import json
import unittest

import nbformat
import pandas as pd

from render_kaplan_meier import (CELL_ID, NOTEBOOK, render, target_cell,
                                 validate_data, with_plot_output)


class KaplanMeierTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"time": [4, 10, 20, 30],
                                  "DEATH_EVENT": [1, 0, 1, 0],
                                  "ejection_fraction": [20, 30, 40, 60]})

    def test_preserves_death_and_censoring_indicators_and_rows(self):
        validated = validate_data(self.data)
        pd.testing.assert_frame_equal(validated, self.data)

    def test_invalid_observations_fail_without_dropping_rows(self):
        cases = [("time", float("nan")), ("time", float("inf")),
                 ("time", 0), ("time", -1), ("DEATH_EVENT", 2),
                 ("DEATH_EVENT", 0.5), ("ejection_fraction", 0),
                 ("ejection_fraction", 101), ("ejection_fraction", "invalid")]
        for column, value in cases:
            with self.subTest(column=column, value=value):
                data = self.data.astype(object)
                data.loc[0, column] = value
                with self.assertRaises(ValueError):
                    validate_data(data)

    def test_empty_or_degenerate_split_fails(self):
        with self.assertRaises(ValueError):
            validate_data(self.data.iloc[:0])
        with self.assertRaises(ValueError):
            validate_data(self.data.assign(ejection_fraction=35))

    def test_existing_plot_cell_renders_synthetic_data(self):
        notebook = json.loads(NOTEBOOK.read_text())
        before = copy.deepcopy(notebook)
        png = render(target_cell(notebook), validate_data(self.data))
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(notebook, before)

    def test_only_target_output_and_execution_stamp_change(self):
        keep = nbformat.v4.new_code_cell("raise RuntimeError('must not execute')", id="keep")
        keep.outputs = [nbformat.v4.new_output("stream", name="stdout", text="saved result")]
        plot = nbformat.v4.new_code_cell("# existing plot source", id=CELL_ID)
        plot.metadata = {"execution": {"old": "timestamp"}, "keep": "metadata"}
        plot.execution_count = 14
        notebook = nbformat.v4.new_notebook(cells=[keep, plot])
        before = copy.deepcopy(notebook)
        updated = with_plot_output(notebook, b"synthetic png payload")
        self.assertEqual(notebook, before)
        self.assertEqual(updated["cells"][0], before["cells"][0])
        self.assertEqual(updated["metadata"], before["metadata"])
        self.assertEqual(updated["cells"][1]["source"], before["cells"][1]["source"])
        self.assertEqual(updated["cells"][1]["metadata"], {"keep": "metadata"})
        self.assertIsNone(updated["cells"][1]["execution_count"])


if __name__ == "__main__":
    unittest.main()
