
"""Experimental shot-type evidence from 2D wrist movement.

This is not a trained cricket-shot classifier.
It never changes the existing 'Shot Played' label.
"""

import math


class ShotTypeV2:

    VALID_PHASES = {
        "backlift",
        "downswing",
        "impact",
        "follow through",
    }

    def __init__(self, min_visibility=0.40):
        self.min_visibility = min_visibility
        self.arm = None
        self.impact_frame = None
        self.samples = []

    # =====================================================
    # READ A VALID LANDMARK
    # =====================================================

    def _point(self, landmarks, key):

        value = (
            landmarks.get(key)
            if isinstance(landmarks, dict)
            else None
        )

        if not isinstance(value, dict):
            return None

        try:
            x = float(value["x"])
            y = float(value["y"])
            visibility = float(
                value.get("visibility", 1.0)
            )
        except (
            TypeError,
            ValueError,
            KeyError,
            OverflowError,
        ):
            return None

        if not all(
            math.isfinite(v)
            for v in (x, y, visibility)
        ):
            return None

        if visibility < self.min_visibility:
            return None

        return x, y

    # =====================================================
    # COLLECT FRAME MEASUREMENTS
    # =====================================================

    def observe(self, frame_number, report, landmarks):

        if not isinstance(report, dict):
            return

        impact = report.get("impact")

        if (
            isinstance(impact, dict)
            and bool(impact.get("event", False))
            and self.impact_frame is None
        ):
            self.impact_frame = frame_number

        # Never reuse old landmarks on a failed pose frame.
        if not report.get("landmarks_detected"):
            return

        phase = str(
            report.get("phase", "")
        ).strip().lower()

        if phase not in self.VALID_PHASES:
            return

        if not isinstance(landmarks, dict):
            return

        # Choose one arm and avoid artificial arm switching.
        if self.arm is None:

            for arm in ("left", "right"):

                wrist = self._point(
                    landmarks,
                    f"{arm}_wrist"
                )

                shoulder = self._point(
                    landmarks,
                    f"{arm}_shoulder"
                )

                if wrist is not None and shoulder is not None:
                    self.arm = arm
                    break

        if self.arm is None:
            return

        wrist = self._point(
            landmarks,
            f"{self.arm}_wrist"
        )

        shoulder = self._point(
            landmarks,
            f"{self.arm}_shoulder"
        )

        if wrist is None or shoulder is None:
            return

        try:
            frame_number = int(frame_number)
        except (
            TypeError,
            ValueError,
            OverflowError
        ):
            return

        if (
            self.samples
            and frame_number <= self.samples[-1]["frame"]
        ):
            return

        # Shoulder-relative position reduces some
        # whole-body translation effects.
        self.samples.append({
            "frame": frame_number,
            "phase": phase,
            "x": wrist[0] - shoulder[0],
            "y": wrist[1] - shoulder[1],
        })

    # =====================================================
    # PHASE-LOCAL WRIST MOVEMENT
    # =====================================================

    def _motion(self, phase):

        items = [
            sample
            for sample in self.samples
            if sample["phase"] == phase
        ]

        horizontal = 0.0
        vertical = 0.0
        path = 0.0
        pairs = 0

        for first, second in zip(items, items[1:]):

            # Do not bridge missing-frame gaps.
            if (
                second["frame"] - first["frame"]
                != 1
            ):
                continue

            dx = second["x"] - first["x"]
            dy = second["y"] - first["y"]

            distance = math.hypot(dx, dy)

            # Reject unusually large pose/crop jumps.
            if distance > 0.30:
                continue

            horizontal += abs(dx)
            vertical += abs(dy)
            path += distance
            pairs += 1

        return {
            "sample_count": len(items),
            "adjacent_motion_pairs": pairs,
            "absolute_horizontal_motion": round(
                horizontal, 4
            ),
            "absolute_vertical_motion": round(
                vertical, 4
            ),
            "total_motion": round(path, 4),
        }

    # =====================================================
    # GENERATE EXPERIMENTAL SHOT-TYPE EVIDENCE
    # =====================================================

    def summarize(self):

        backlift = self._motion("backlift")
        downswing = self._motion("downswing")
        follow_through = self._motion("follow through")

        limitations = []
        candidate = "None"

        if self.impact_frame is None:
            limitations.append(
                "No confirmed impact event observed."
            )

        if (
            downswing["sample_count"] < 4
            or downswing["adjacent_motion_pairs"] < 3
        ):
            limitations.append(
                "Insufficient consecutive Downswing "
                "wrist measurements."
            )

        if (
            follow_through["sample_count"] < 4
            or follow_through["adjacent_motion_pairs"] < 3
        ):
            limitations.append(
                "Insufficient consecutive Follow Through "
                "wrist measurements."
            )

        if not limitations:

            horizontal = downswing[
                "absolute_horizontal_motion"
            ]

            vertical = downswing[
                "absolute_vertical_motion"
            ]

            follow_path = follow_through[
                "total_motion"
            ]

            if max(horizontal, vertical) < 0.08:

                limitations.append(
                    "Pre-impact wrist movement is too "
                    "small to characterize."
                )

            elif horizontal >= 1.6 * max(vertical, 0.001):

                candidate = "Cross-bat-like wrist movement"

                limitations.append(
                    "Cannot distinguish Cut from Pull "
                    "without reliable contact height, "
                    "shot side and camera orientation."
                )

            elif vertical >= 1.6 * max(horizontal, 0.001):

                if follow_path >= 0.12:

                    candidate = "Drive-like wrist movement"

                elif follow_path <= 0.04:

                    candidate = (
                        "Defensive-like wrist movement"
                    )

                else:

                    candidate = (
                        "Vertical-bat-like wrist movement"
                    )

                    limitations.append(
                        "Follow-through size does not "
                        "separate Drive and Defense."
                    )

            else:

                candidate = "Mixed wrist movement"

                limitations.append(
                    "Wrist trajectory has no dominant "
                    "movement plane."
                )

        return {
            "status": (
                "Experimental candidate"
                if candidate != "None"
                else "Insufficient evidence"
            ),
            "shot_type": "Uncertain",
            "motion_candidate": candidate,
            "impact_frame": self.impact_frame,
            "tracked_arm": self.arm,
            "evidence": {
                "backlift": backlift,
                "downswing": downswing,
                "follow_through": follow_through,
            },
            "limitations": limitations,
            "note": (
                "Unverified 2D wrist-motion hypothesis, "
                "not a trained shot-type prediction or "
                "probability. Wrist motion alone cannot "
                "reliably identify Drive, Cut, Pull or "
                "Defense across camera views. Existing "
                "shot completion and scores are unchanged."
            ),
        }
