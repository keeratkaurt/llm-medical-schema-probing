import csv
import unittest
from pathlib import Path

from analyze import analyze


class AnalysisTest(unittest.TestCase):
    def test_checked_in_results_have_expected_size(self):
        for model in ("gpt2", "distilgpt2"):
            with self.subTest(model=model):
                with (Path("results") / model / "scores.csv").open(newline="") as stream:
                    report = analyze(csv.DictReader(stream))
                self.assertEqual(report["n_phrases"], 24)
                self.assertEqual(report["n_prompts"], 2)
                self.assertEqual(sum(len(report["confusion"][p][d])
                                     for p in report["confusion"] for d in report["confusion"][p]), 32)

    def test_inconsistent_prediction_is_rejected(self):
        with Path("results/gpt2/scores.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["predicted"] = "celiac disease"
        with self.assertRaises(ValueError):
            analyze(rows)


if __name__ == "__main__":
    unittest.main()
