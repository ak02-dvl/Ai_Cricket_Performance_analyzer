"""Run from project root: python -m unittest regression.test_score_evidence -v"""
import unittest
from analysis.score_evidence import build_score_evidence


class ScoreEvidenceTests(unittest.TestCase):
    def test_vk_partial_coverage(self):
        result = build_score_evidence({
            "overall_score": 72.7,
            "scores": {
                "left_elbow": 20.2, "left_knee": 60.5,
                "right_knee": 100, "left_shoulder": 100,
                "left_hip": 90.9, "right_hip": 64.7,
            },
            "joint_measurement_reliability": {
                "measured_joints": 6,
                "missing_joints": ["right_elbow", "right_shoulder"],
            },
        })
        self.assertEqual(result["status"], "Partial joint evidence")
        self.assertEqual(result["score_input_coverage_percent"], 75.0)
        self.assertEqual(result["reported_score_unchanged"], 72.7)
        self.assertEqual(
            result["illustrative_full_joint_average_range"],
            {"minimum": 54.5, "maximum": 79.5},
        )
        self.assertEqual(result["warnings"], [])

    def test_rohit_partial_coverage(self):
        result = build_score_evidence({
            "overall_score": 70.3,
            "scores": {
                "left_elbow": 8.5, "left_knee": 61,
                "right_knee": 100, "left_shoulder": 100,
                "left_hip": 79.1, "right_hip": 73.2,
            },
            "joint_measurement_reliability": {
                "measured_joints": 6,
                "missing_joints": ["right_elbow", "right_shoulder"],
            },
        })
        self.assertEqual(result["status"], "Partial joint evidence")
        self.assertEqual(result["score_input_coverage_percent"], 75.0)
        self.assertEqual(result["reported_score_unchanged"], 70.3)
        self.assertEqual(result["warnings"], [])

    def test_no_coverage_is_opt_in(self):
        self.assertIsNone(build_score_evidence({"scores": {"left_knee": 100}}))

    def test_inconsistent_data_is_flagged(self):
        result = build_score_evidence({
            "scores": {"left_knee": 100},
            "joint_measurement_reliability": {
                "measured_joints": 2, "missing_joints": [],
            },
        })
        self.assertEqual(result["status"], "Check score inputs")
        self.assertTrue(result["warnings"])


if __name__ == "__main__":
    unittest.main()
