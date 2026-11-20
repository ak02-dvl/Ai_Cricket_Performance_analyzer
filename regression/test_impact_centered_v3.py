import copy
import unittest
from analysis.impact_centered_v3 import ImpactCenteredMotionV3


def pose(x, y, visibility=0.95):
    return {
        'left_shoulder': {'x': 0.5, 'y': 0.4, 'visibility': 0.95},
        'left_wrist': {'x': x, 'y': y, 'visibility': visibility},
    }


def report(phase='Backlift', impact=False, pose_detected=True, ball=None):
    result = {
        'phase': phase,
        'shot': 'Shot Played' if impact else 'Controlled Shot',
        'overall_score': 70.3,
        'landmarks_detected': pose_detected,
        'ball_detected': ball is not None,
        'ball': ball,
        'impact': {'event': impact},
    }
    if impact:
        result['impact'].update({
            'impact_frame': 123,
            'overlap': True,
            'bat_input': {'accepted': True, 'box': [160, 425, 235, 660]},
            'ball_input': {'accepted': True, 'box': [146, 592, 170, 621]},
        })
    return result


class ImpactCenteredMotionTests(unittest.TestCase):
    def test_without_impact_is_uncertain(self):
        tracker = ImpactCenteredMotionV3()
        for f in range(105, 110):
            tracker.observe(f, report(), pose(0.5 + (f-105)*0.04, 0.45))
        self.assertEqual(tracker.summarize()['status'], 'No confirmed impact')

    def test_rohit_one_downswing_frame_can_use_preimpact_motion(self):
        tracker = ImpactCenteredMotionV3()
        # pre-impact frames, differing phase labels, and a 3-frame landmark gap
        for f in range(110, 123):
            if 118 <= f <= 120:
                tracker.observe(f, report(pose_detected=False), pose(0.1, 0.2))
            else:
                tracker.observe(f, report('Downswing' if f == 122 else 'Backlift'), pose(0.45 + .022*(f-110), 0.45))
        impact_report = report('Impact', impact=True)
        snapshot = copy.deepcopy(impact_report)
        tracker.observe(123, impact_report, pose(0.8, 0.4))
        self.assertEqual(snapshot, impact_report)
        for f in range(124, 129):
            tracker.observe(f, report('Follow Through'), pose(0.6, 0.4 + .012*(f-124)))
        s = tracker.summarize()
        self.assertEqual(s['impact_frame'], 123)
        self.assertEqual(s['verified_shot_type'], 'Uncertain')
        self.assertNotEqual(s['motion_candidate'], 'None')
        self.assertGreaterEqual(s['wrist_evidence']['before_impact']['adjacent_motion_pairs'], 3)
        self.assertIn('Missing pre-impact pose frames', ' '.join(s['limitations']))
        self.assertEqual(s['contact_evidence']['bat_box_axis'], 'Tall rectangle')

    def test_stance_frames_included_without_phase_filtering(self):
        tracker = ImpactCenteredMotionV3()
        for f in range(22, 35):
            tracker.observe(f, report('Stance' if f < 29 else 'Backlift'), pose(0.5, 0.45 + 0.02*(f-22)))
        tracker.observe(35, report('Impact', impact=True), pose(0.5, 0.7))
        s = tracker.summarize()
        self.assertGreaterEqual(s['wrist_evidence']['before_impact']['sample_count'], 10)
        self.assertEqual(s['motion_candidate'], 'Vertical-dominant pre-impact wrist movement')

    def test_stale_pose_is_ignored_and_gap_not_bridged(self):
        tracker = ImpactCenteredMotionV3()
        tracker.observe(1, report(pose_detected=True), pose(0.5, 0.5))
        tracker.observe(2, report(pose_detected=False), pose(0.6, 0.6))
        tracker.observe(3, report(pose_detected=True), pose(0.7, 0.7))
        tracker.observe(4, report('Impact', impact=True), pose(0.7, 0.7))
        summary = tracker.summarize()
        self.assertEqual(summary['wrist_evidence']['before_impact']['adjacent_motion_pairs'], 0)

    def test_predicted_ball_excluded_and_does_not_bridge_gaps(self):
        tracker = ImpactCenteredMotionV3()
        real = lambda x: {'center': [x, 10], 'predicted': False}
        tracker.observe(20, report(ball=real(10)), None)
        tracker.observe(21, report(ball=real(20)), None)
        tracker.observe(22, report(ball={'center': [30, 10], 'predicted': True}), None)
        tracker.observe(23, report(ball=real(40)), None)
        tracker.observe(24, report('Impact', impact=True), None)
        tracker.observe(25, report(ball=real(50)), None)
        tracker.observe(26, report(ball=real(60)), None)
        summary = tracker.summarize()
        self.assertEqual(summary['ball_evidence']['before_impact']['adjacent_motion_pairs'], 1)
        self.assertNotEqual(summary['ball_evidence']['status'], 'Pre- and post-impact ball displacement observed')

    def test_invalid_geometry_does_not_claim_bat_orientation(self):
        tracker = ImpactCenteredMotionV3()
        r = report('Impact', impact=True)
        r['impact']['bat_input']['box'] = [2, 3, 1, 4]
        tracker.observe(123, r, pose(0.5, 0.6))
        self.assertIsNone(tracker.summarize()['contact_evidence']['bat_box_axis'])


if __name__ == '__main__':
    unittest.main()
