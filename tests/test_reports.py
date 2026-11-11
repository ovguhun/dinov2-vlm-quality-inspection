"""Tests for the report scripts: the parser of 07 and the summaries of 08.

Run from the repository root: python -m unittest discover tests
"""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

REPORT = {
    "visual_evidence": "A dark scratch near the top.",
    "inspection_status": "anomalous",
    "defect_type": "scratch",
    "location": "top",
    "severity_estimate": "limited",
    "recommended_action": "Inspect the scratch closely.",
}


def load_script(name):
    """A script in scripts/, imported as a module."""
    spec = importlib.util.spec_from_file_location(
        name, Path("scripts") / f"{name}.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


class ParseReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script("07_generate_reports")
        with open("configs/report_schema.json", encoding="utf-8") as file:
            cls.schema = json.load(file)

    def parse(self, answer):
        return self.script.parse_report(answer, self.schema)

    def test_valid_answers(self):
        text = json.dumps(REPORT)
        for answer in (text, f"```json\n{text}\n```", f"```\n{text}\n```"):
            self.assertEqual(self.parse(answer), (REPORT, ""))

    def test_problems(self):
        text = json.dumps(REPORT)
        problems = {
            "": "empty answer",
            text[:-1] + ', "location": "left"}': "duplicate key",
            text.replace('"limited"', "NaN"): "not a JSON number",
            text.replace('"limited"', "1e999"): "number out of range",
            "[]": "is not of type 'object'",
            text.replace('"anomalous"', '"Anomalous"'): "is not one of",
            text[:-1] + ', "score": 0.9}': "Additional properties",
        }
        for answer, problem in problems.items():
            self.assertIn(problem, self.parse(answer)[1], answer)


class EvaluateReports(unittest.TestCase):
    """08 on copies of results/reports, in a temporary folder."""

    @classmethod
    def setUpClass(cls):
        cls.script = load_script("08_evaluate_reports")
        cls.results = Path("results/reports").resolve()
        cls.config = Path("configs/reporting.json").resolve()

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(temp.name)
        Path("configs").mkdir()
        shutil.copy(self.config, "configs")
        Path("outputs/reports").mkdir(parents=True)
        for name in ("cases.csv", "reports.csv"):
            shutil.copy(self.results / name, "outputs/reports")

    def run_script(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.script.main()

    def test_thesis_grades_give_the_published_summaries(self):
        shutil.copy(self.results / "grades.csv", "outputs/reports")
        self.run_script()
        for name in ("summary.csv", "paired_changes.csv"):
            self.assertEqual(Path("outputs/reports", name).read_bytes(),
                             (self.results / name).read_bytes(), name)

    def test_a_blank_sheet_comes_first(self):
        self.run_script()
        self.assertTrue(Path("outputs/reports/grades.csv").is_file())
        with self.assertRaises(SystemExit):
            self.run_script()  # the sheet is still empty

    def test_broken_sheets_stop_with_a_message(self):
        """Each broken copy of the thesis grades stops 08 at its own check."""
        thesis = self.script.read_csv(self.results / "grades.csv")

        def changed(case_id, condition, column, grade):
            rows = [dict(row) for row in thesis]
            for row in rows:
                if (row["case_id"], row["condition"]) == (case_id, condition):
                    row[column] = grade
            return rows

        sheets = {
            "case-008 D0: detector_overlay should not be n/a": changed(
                "case-008", "D0", "detector_overlay", "n/a"),
            "case-008 I0: location_grade is 'appropriate n/a'": changed(
                "case-008", "I0", "location_grade", "appropriate n/a"),
            "case-040: 'reference unavailable' must be given": changed(
                "case-040", "D1", "location_grade", "not established"),
            "case-008 D0: the grades were given for a different answer": changed(
                "case-008", "D0", "answer_digest", "0" * 12),
            "needs exactly one row per report": thesis + thesis[:1],
            "lacks contradiction": [
                {k: v for k, v in row.items() if k != "contradiction"}
                for row in thesis],
        }
        for message, sheet in sheets.items():
            self.script.write_csv(Path("outputs/reports/grades.csv"), sheet)
            with self.assertRaises(SystemExit, msg=message) as stop:
                self.run_script()
            self.assertIn(message, str(stop.exception))


if __name__ == "__main__":
    unittest.main()
