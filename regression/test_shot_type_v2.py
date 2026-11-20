
import unittest

from analysis.shot_type_v2 import ShotTypeV2


def landmarks(x, y):
    return {
        "left_wrist": {
            "x": x,
            "y": y,
            "visibility": 0.95
        },
        "left_shoulder": {
            "x": 0.5,
            "y": 0.4,
            "visibility": 0.95
        },
    }


def feed(
    tracker,
    down_step=(0.001, 0.04),
    follow_step=(0.001, 0.04),
    impact=True
):
    x, y = 0.5, 0.4

    for frame in range(1, 6):
        x += down_step[0]
        y += down_step[1]

        tracker.observe(
            frame,
            {
                "phase": "Downswing",
                "landmarks_detected": True
            },
            landmarks(x, y)
        )

    tracker.observe(
        6,
        {
            "phase": "Impact",
            "landmarks_detected": True,
            "impact": {"event": impact}
        },
        landmarks(x, y)
    )

    for frame in range(7, 12):
        x += follow_step[0]
        y += follow_step[1]

        tracker.observe(
            frame,
            {
                "phase": "Follow Through",
                "landmarks_detected": True
            },
            landmarks(x, y)
        )


class ShotTypeV2Tests(unittest.TestCase):

    def test_no_impact(self):
        tracker = ShotTypeV2()
        feed(tracker, impact=False)

        result = tracker.summarize()

        self.assertEqual(
            result["shot_type"],
            "Uncertain"
        )

        self.assertEqual(
            result["motion_candidate"],
            "None"
        )

    def test_drive_like_movement(self):
        tracker = ShotTypeV2()
        feed(tracker)

        result = tracker.summarize()

        self.assertEqual(
            result["motion_candidate"],
            "Drive-like wrist movement"
        )

        self.assertEqual(
            result["shot_type"],
            "Uncertain"
        )

    def test_cross_bat_does_not_guess_cut_or_pull(self):
        tracker = ShotTypeV2()

        feed(
            tracker,
            down_step=(0.045, 0.002)
        )

        result = tracker.summarize()

        self.assertEqual(
            result["motion_candidate"],
            "Cross-bat-like wrist movement"
        )

        self.assertEqual(
            result["shot_type"],
            "Uncertain"
        )

    def test_insufficient_downswing(self):
        tracker = ShotTypeV2()

        tracker.observe(
            122,
            {
                "phase": "Downswing",
                "landmarks_detected": True
            },
            landmarks(0.6, 0.6)
        )

        tracker.observe(
            123,
            {
                "phase": "Impact",
                "landmarks_detected": True,
                "impact": {"event": True}
            },
            landmarks(0.6, 0.6)
        )

        result = tracker.summarize()

        self.assertEqual(
            result["motion_candidate"],
            "None"
        )

    def test_reject_stale_landmarks(self):
        tracker = ShotTypeV2()

        tracker.observe(
            1,
            {
                "phase": "Downswing",
                "landmarks_detected": False
            },
            landmarks(0.6, 0.6)
        )

        self.assertEqual(tracker.samples, [])

    def test_original_report_is_unchanged(self):
        tracker = ShotTypeV2()

        report = {
            "phase": "Impact",
            "shot": "Shot Played",
            "overall_score": 70.3,
            "landmarks_detected": True,
            "impact": {"event": True}
        }

        original = {
            **report,
            "impact": dict(report["impact"])
        }

        tracker.observe(
            123,
            report,
            landmarks(0.6, 0.6)
        )

        self.assertEqual(report, original)


if __name__ == "__main__":
    unittest.main()
