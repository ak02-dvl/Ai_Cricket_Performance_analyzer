import copy
import math
import unittest
from analysis.tracker_association_v34 import TrackerAssociationV34


def audit(rows, impact=10):
    return {"impact_frame": impact, "frame_evidence": rows}


def point(frame, raw=None, tracked=None, source="fresh_raw_ball"):
    return {"frame": frame, "raw_ball_center": raw,
            "tracked_ball_center": tracked, "ball_source": source}


class TrackerAssociationV34Tests(unittest.TestCase):
    def setUp(self):
        self.audit_tool = TrackerAssociationV34(476, 848)

    def test_missing_audit(self):
        result = self.audit_tool.summarize(None)
        self.assertEqual(result["status"], "Unavailable")

    def test_missing_impact(self):
        self.assertEqual(self.audit_tool.summarize(audit([], None))["status"], "Unavailable")

    def test_normal_alignment(self):
        result = self.audit_tool.summarize(audit([
            point(9, [40, 50], [42, 51]), point(10, [50, 60], [52, 62]),
            point(11, [60, 70], [65, 73])
        ]))
        self.assertEqual(result["issue_count"], 0)

    def test_rohit_style_disagreement(self):
        result = self.audit_tool.summarize(audit([
            point(10, [158, 606], [159, 625]),
            point(11, None, [157, 560], "prediction_only_no_raw_ball"),
            point(12, [378, 616], [156, 514]),
        ]))
        self.assertIn("raw_tracker_disagreement", [x["kind"] for x in result["issues"]])
        self.assertEqual(result["windows"]["after"]["large_raw_tracker_gap"], 1)

    def test_out_of_frame_tracker(self):
        result = TrackerAssociationV34(1920, 1080).summarize(audit([
            point(10, [1116, 500], [1120, 490]),
            point(11, [734, 509], [811, 1110])
        ]))
        self.assertIn("tracked_out_of_frame", [x["kind"] for x in result["issues"]])

    def test_raw_candidate_without_tracker(self):
        result = self.audit_tool.summarize(audit([point(11, [80, 90], None)]))
        self.assertEqual(result["windows"]["after"]["raw_without_tracker_position"], 1)

    def test_no_connect_across_gaps(self):
        result = self.audit_tool.summarize(audit([
            point(8, [1, 1], [1, 1]), point(11, [400, 400], [400, 400])
        ]))
        self.assertNotIn("large_consecutive_raw_step", [x["kind"] for x in result["issues"]])

    def test_large_adjacent_raw_step(self):
        result = self.audit_tool.summarize(audit([
            point(10, [10, 10], [10, 10]), point(11, [350, 350], [350, 350])
        ]))
        self.assertIn("large_consecutive_raw_step", [x["kind"] for x in result["issues"]])

    def test_invalid_coordinates_ignored(self):
        result = self.audit_tool.summarize(audit([
            point(10, [math.nan, 1], [math.inf, 2]),
            point(11, "bad", [1, 2])
        ]))
        self.assertEqual(result["issue_count"], 0)

    def test_original_audit_unchanged(self):
        original = audit([point(10, [10, 20], [12, 22]), point(11, [40, 40], [42, 43])])
        before = copy.deepcopy(original)
        self.audit_tool.summarize(original)
        self.assertEqual(before, original)

    def test_frame_out_of_order_ignored(self):
        result = self.audit_tool.summarize(audit([
            point(10, [10, 10], [10, 10]), point(10, [400, 400], [400, 400]),
            point(11, [11, 11], [11, 11])
        ]))
        self.assertEqual(result["windows"]["impact"]["observed_frames"], 1)

    def test_frame_dimensions_guard(self):
        with self.assertRaises(ValueError):
            TrackerAssociationV34(0, 848)


if __name__ == "__main__":
    unittest.main()
