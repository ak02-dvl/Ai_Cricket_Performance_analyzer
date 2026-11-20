"""Self-tests for the JSON regression checker (no video or ML dependencies)."""

import copy
import json
import unittest
from pathlib import Path

from check_regression import check_report

HERE = Path(__file__).resolve().parent


class RegressionCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baselines = json.loads((HERE / "baselines.json").read_text(encoding="utf-8"))["videos"]
        cls.reports = {
            "vk.mp4": json.loads((HERE / "samples/vk_reference.json").read_text(encoding="utf-8")),
            "rohit.mp4": json.loads((HERE / "samples/rohit_reference.json").read_text(encoding="utf-8")),
        }

    def test_known_good_reference_reports_pass(self):
        for name in self.reports:
            with self.subTest(video=name):
                self.assertEqual(check_report(self.reports[name], self.baselines[name]), [])

    def test_catches_shifted_impact_frame(self):
        data = copy.deepcopy(self.reports["vk.mp4"])
        data["impact_events"][0]["impact_frame"] = 36
        errors = check_report(data, self.baselines["vk.mp4"])
        self.assertTrue(any("impact_events.0.impact_frame" in item for item in errors))

    def test_catches_score_change(self):
        data = copy.deepcopy(self.reports["rohit.mp4"])
        data["final_score"] = 52.0
        errors = check_report(data, self.baselines["rohit.mp4"])
        self.assertTrue(any("final_score" in item for item in errors))

    def test_catches_bad_coverage_percent(self):
        data = copy.deepcopy(self.reports["rohit.mp4"])
        data["swing_summary"]["phase_coverage"]["phases"]["downswing"]["motion_sample_coverage_percent"] = 100.0
        errors = check_report(data, self.baselines["rohit.mp4"])
        self.assertTrue(any("motion_sample_coverage_percent" in item for item in errors))

    def test_catches_missing_coverage(self):
        data = copy.deepcopy(self.reports["vk.mp4"])
        del data["swing_summary"]["phase_coverage"]
        errors = check_report(data, self.baselines["vk.mp4"])
        self.assertTrue(any("phase_coverage" in item for item in errors))

    def test_catches_inconsistent_motion_sample_totals(self):
        data = copy.deepcopy(self.reports["vk.mp4"])
        data["swing_summary"]["samples"] = 29
        errors = check_report(data, self.baselines["vk.mp4"])
        self.assertTrue(any("swing sample total" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
