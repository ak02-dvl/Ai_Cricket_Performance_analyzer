import copy
import unittest

from analysis.bat_ball_motion_v32 import BatBallMotionV32


def rep(ball=None, impact=False):
    return {
        'analysis_status': 'Analysis Working',
        'phase': 'Backlift', 'shot': 'Shot Played', 'overall_score': 70.3,
        'ball_detected': ball is not None, 'ball': ball,
        'impact': ({'event': True, 'impact_frame': 20, 'overlap': True,
                    'bat_input': {'accepted': True, 'box': [30, 30, 40, 120]},
                    'ball_input': {'accepted': True, 'box': [40, 60, 50, 70]}}
                   if impact else {'event': False}),
    }


def ball(x, y, *, predicted=False, missing=0, confidence=.8):
    return {'center': [x, y], 'predicted': predicted,
            'missing_frames': missing, 'confidence': confidence}


def bat(x, y, fresh=True):
    return {'fresh': fresh, 'box': [x, y, x+20, y+40], 'confidence': .5}


class BatBallMotionV32Tests(unittest.TestCase):
    def test_no_impact_returns_uncertain(self):
        tr = BatBallMotionV32(640, 360)
        tr.observe(1, rep(), bat(10, 10))
        self.assertEqual(tr.summarize()['status'], 'No confirmed impact')
        self.assertEqual(tr.summarize()['verified_shot_type'], 'Uncertain')

    def test_two_sided_ball_turn_is_diagnostic_only(self):
        tr = BatBallMotionV32(640, 360)
        for f in range(16, 20):
            tr.observe(f, rep(ball(70 + 7*(f-16), 90)), bat(10+3*f, 10))
        snap = rep(impact=True)
        unchanged = copy.deepcopy(snap)
        tr.observe(20, snap, bat(70, 10))
        self.assertEqual(snap, unchanged)
        for f in range(21, 25):
            tr.observe(f, rep(ball(100, 90 - 8*(f-21))), bat(70+3*(f-20), 10))
        result = tr.summarize()
        self.assertEqual(result['status'], 'Trajectory comparison available')
        self.assertEqual(result['verified_shot_type'], 'Uncertain')
        self.assertAlmostEqual(result['ball_motion']['screen_direction_change_degrees'], 90.0)
        self.assertEqual(result['impact_frame'], 20)

    def test_rohit_missing_postimpact_ball_no_outgoing_claim(self):
        tr = BatBallMotionV32(476, 848)
        for f in range(8, 20):
            tr.observe(f, rep(ball(f*3, 100+f*2)), bat(5*f, 80))
        tr.observe(20, rep(impact=True))
        for f in range(21, 28):
            tr.observe(f, rep(), bat(5*f, 80))
        result = tr.summarize()
        self.assertIsNone(result['ball_motion']['screen_direction_change_degrees'])
        self.assertEqual(result['ball_motion']['trajectory_status'], 'Not verified across impact')
        self.assertEqual(result['ball_motion']['after_impact']['samples'], 0)

    def test_predictions_low_confidence_and_tracker_bat_rejected(self):
        tr = BatBallMotionV32(640, 360)
        tr.observe(17, rep(ball(1, 2)), bat(10, 20, fresh=False))
        tr.observe(18, rep(ball(2, 4, predicted=True)), bat(20, 30))
        tr.observe(19, rep(ball(3, 6, confidence=.01)), bat(30, 40, fresh=False))
        tr.observe(20, rep(impact=True))
        result = tr.summarize()
        self.assertEqual(result['bat_box_motion']['before_impact']['samples'], 1)
        self.assertEqual(result['ball_motion']['before_impact']['samples'], 1)

    def test_gaps_do_not_count_as_motion_pairs(self):
        tr = BatBallMotionV32(640, 360)
        tr.observe(17, rep(ball(10, 30)))
        tr.observe(19, rep(ball(90, 30)))
        tr.observe(20, rep(impact=True))
        s = tr.summarize()
        self.assertEqual(s['ball_motion']['before_impact']['adjacent_pairs'], 0)
        self.assertEqual(s['ball_motion']['before_impact']['status'], 'Limited evidence')

    def test_invalid_boxes_and_duplicate_frames_rejected(self):
        tr = BatBallMotionV32(640, 360)
        tr.observe(17, rep(), {'fresh': True, 'confidence': .9, 'box': [50, 50, 20, 20]})
        tr.observe(17, rep(ball(20, 20)), bat(0, 0))
        tr.observe(20, rep(impact=True))
        result = tr.summarize()
        self.assertEqual(result['bat_box_motion']['before_impact']['samples'], 0)
        self.assertEqual(result['ball_motion']['before_impact']['samples'], 0)

    def test_per_video_isolation(self):
        a = BatBallMotionV32(640, 360)
        b = BatBallMotionV32(640, 360)
        a.observe(20, rep(impact=True))
        self.assertIsNone(b.summarize()['impact_frame'])

    def test_large_tracking_jumps_rejected(self):
        tr = BatBallMotionV32(640, 360)
        tr.observe(17, rep(ball(10, 10)))
        tr.observe(18, rep(ball(600, 300)))
        tr.observe(20, rep(impact=True))
        result = tr.summarize()
        self.assertEqual(result['ball_motion']['before_impact']['rejected_jumps'], 1)
        self.assertEqual(result['ball_motion']['before_impact']['adjacent_pairs'], 0)


if __name__ == '__main__':
    unittest.main()
