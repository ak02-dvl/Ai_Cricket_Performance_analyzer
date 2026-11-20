import unittest
from analysis.detection_evidence_v33 import DetectionEvidenceV33


def snapshot(frame, bat=None, raw=None, tracked=None):
    return {"frame": frame, "stage": "after_ball_tracker", "bat_candidate": bat,
            "raw_ball": raw, "tracked_ball": tracked}


def bat(x=10, tracker_only=False):
    return {"box": [x, 10, x + 7, 45], "confidence": .80,
            "tracker_only": tracker_only}


def ball(x=10, predicted=False, confidence=.90):
    return {"box": [x, 20, x + 4, 24], "center": [x + 2, 22],
            "confidence": confidence, "predicted": predicted, "missing_frames": 0}


def impact_report(frame):
    return {"impact": {"event": True, "impact_frame": frame},
            "analysis_status": "Analysis Working"}


class DetectionEvidenceTests(unittest.TestCase):
    def test_raw_and_tracker_sources_distinguished(self):
        a = DetectionEvidenceV33()
        a.observe(10, {}, snapshot(10, bat(), ball(), ball()))
        a.observe(11, impact_report(11), snapshot(11, bat(), ball(), ball()))
        a.observe(12, {}, snapshot(12, bat(15, True), None, ball(predicted=True)))
        x = a.summarize()
        self.assertEqual(x['before_impact']['fresh_raw_ball_frames'], 1)
        self.assertEqual(x['after_impact']['prediction_only_ball_frames'], 1)
        self.assertEqual(x['after_impact']['csrt_only_frames'], 1)

    def test_tracked_nonpred_without_raw_is_not_fresh(self):
        a = DetectionEvidenceV33()
        a.observe(1, impact_report(1), snapshot(1))
        a.observe(2, {}, snapshot(2, None, None, ball(predicted=False)))
        x = a.summarize()
        self.assertEqual(x['after_impact']['fresh_raw_ball_frames'], 0)
        self.assertEqual(x['after_impact']['tracked_without_raw_frames'], 1)

    def test_no_raw_or_tracker_is_missing(self):
        a = DetectionEvidenceV33()
        a.observe(1, impact_report(1), snapshot(1))
        a.observe(2, {}, snapshot(2))
        self.assertEqual(a.summarize()['after_impact']['missing_raw_and_tracked_ball_frames'], 1)

    def test_missing_stage_not_misreported_as_no_ball(self):
        a = DetectionEvidenceV33()
        a.observe(1, impact_report(1), snapshot(1))
        a.observe(2, {'analysis_status':'Batsman Not Detected'}, None)
        x = a.summarize()['after_impact']
        self.assertEqual(x['stage_not_reached_frames'], 1)
        self.assertEqual(x['missing_raw_and_tracked_ball_frames'], 0)

    def test_low_conf_raw_never_qualifies(self):
        a = DetectionEvidenceV33()
        a.observe(1, impact_report(1), snapshot(1))
        a.observe(2, {}, snapshot(2, None, ball(confidence=.05)))
        self.assertEqual(a.summarize()['after_impact']['fresh_raw_ball_frames'], 0)

    def test_duplicate_frame_ignored(self):
        a = DetectionEvidenceV33()
        a.observe(1, impact_report(1), snapshot(1))
        a.observe(2, {}, snapshot(2, None, ball()))
        a.observe(2, {}, snapshot(2, None, ball()))
        self.assertEqual(a.summarize()['after_impact']['observed_frames'], 1)

    def test_static_candidate_center_flagged_not_declared_stale(self):
        a = DetectionEvidenceV33()
        a.observe(1, impact_report(1), snapshot(1))
        a.observe(2, {}, snapshot(2, bat()))
        a.observe(3, {}, snapshot(3, bat()))
        x = a.summarize()
        self.assertEqual(x['after_impact']['repeated_selected_bat_center_pairs'], 1)
        self.assertTrue(any('verify on video' in v for v in x['limitations']))

    def test_changing_bat_centers_not_static(self):
        a = DetectionEvidenceV33()
        a.observe(1, impact_report(1), snapshot(1))
        a.observe(2, {}, snapshot(2, bat(10)))
        a.observe(3, {}, snapshot(3, bat(14)))
        self.assertEqual(a.summarize()['after_impact']['repeated_selected_bat_center_pairs'], 0)

    def test_original_report_untouched(self):
        a = DetectionEvidenceV33()
        report = impact_report(1)
        snap = snapshot(1, bat(), ball(), ball())
        saved = repr((report, snap))
        a.observe(1, report, snap)
        self.assertEqual(repr((report, snap)), saved)

    def test_no_impact_is_uncertain(self):
        a = DetectionEvidenceV33()
        a.observe(1, {}, snapshot(1, bat(), ball()))
        self.assertEqual(a.summarize()['verified_shot_type'], 'Uncertain')
        self.assertEqual(a.summarize()['impact_frame'], None)

    def test_per_video_state_isolated(self):
        a, b = DetectionEvidenceV33(), DetectionEvidenceV33()
        a.observe(35, impact_report(35), snapshot(35))
        b.observe(123, impact_report(123), snapshot(123))
        self.assertEqual(a.summarize()['impact_frame'],35)
        self.assertEqual(b.summarize()['impact_frame'],123)

    def test_invalid_geometry_not_accepted(self):
        a = DetectionEvidenceV33()
        a.observe(1, impact_report(1), snapshot(1))
        a.observe(2, {}, snapshot(2,
                    {'box':[0,0,0,0],'confidence':.8},
                    {'box':[0,0,0,0],'confidence':.8}))
        x=a.summarize()['after_impact']
        self.assertEqual(x['selected_bat_candidate_frames'],0)
        self.assertEqual(x['fresh_raw_ball_frames'],0)


if __name__ == '__main__':
    unittest.main()
