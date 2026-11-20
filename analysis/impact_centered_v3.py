
"""Read-only impact-centered cricket motion analysis."""

import math
from collections import deque


class ImpactCenteredMotionV3:

    def __init__(
        self,
        pre_frames=14,
        post_frames=12,
        min_visibility=0.40
    ):
        self.pre_frames = max(4, int(pre_frames))
        self.post_frames = max(4, int(post_frames))
        self.min_visibility = float(min_visibility)

        self.arm = None
        self.impact_frame = None
        self.wrist_samples = deque(maxlen=600)
        self.ball_samples = deque(maxlen=600)
        self.contact_evidence = None
        self.last_frame = 0

    # =====================================================
    # LANDMARK VALIDATION
    # =====================================================

    def _point(self, landmarks, key):

        point = (
            landmarks.get(key)
            if isinstance(landmarks, dict)
            else None
        )

        if not isinstance(point, dict):
            return None

        try:
            x = float(point["x"])
            y = float(point["y"])
            visibility = float(
                point.get("visibility", 1.0)
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError
        ):
            return None

        if not all(
            math.isfinite(value)
            for value in (x, y, visibility)
        ):
            return None

        if visibility < self.min_visibility:
            return None

        return x, y

    # =====================================================
    # VALIDATE DETECTION BOX
    # =====================================================

    @staticmethod
    def _box(value):

        if (
            not isinstance(value, (tuple, list))
            or len(value) != 4
        ):
            return None

        try:
            x1, y1, x2, y2 = map(float, value)
        except (
            TypeError,
            ValueError,
            OverflowError
        ):
            return None

        if not all(
            math.isfinite(v)
            for v in (x1, y1, x2, y2)
        ):
            return None

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2

    # =====================================================
    # IMPACT CONTACT GEOMETRY
    # =====================================================

    def _contact_snapshot(self, impact):

        bat = impact.get("bat_input")
        ball = impact.get("ball_input")

        bat_box = (
            self._box(bat.get("box"))
            if isinstance(bat, dict)
            and bat.get("accepted") is True
            else None
        )

        ball_box = (
            self._box(ball.get("box"))
            if isinstance(ball, dict)
            and ball.get("accepted") is True
            else None
        )

        if bat_box is None or ball_box is None:
            return {
                "status": "Fresh bat/ball boxes unavailable",
                "bat_box_axis": None,
                "overlap_reported": bool(
                    impact.get("overlap", False)
                ),
            }

        width = bat_box[2] - bat_box[0]
        height = bat_box[3] - bat_box[1]
        ratio = height / width

        if ratio > 1.25:
            axis = "Tall rectangle"
        elif ratio < 0.8:
            axis = "Wide rectangle"
        else:
            axis = "Near-square rectangle"

        return {
            "status": "Fresh bat and ball boxes reported",
            "bat_box_axis": axis,
            "bat_box_aspect_ratio": round(ratio, 3),
            "overlap_reported": bool(
                impact.get("overlap", False)
            ),
            "ball_box_center_px": [
                round(
                    (ball_box[0] + ball_box[2]) / 2,
                    1
                ),
                round(
                    (ball_box[1] + ball_box[3]) / 2,
                    1
                ),
            ],
            "note": (
                "Axis-aligned detection box shape is "
                "NOT the bat orientation or swing angle."
            ),
        }

    # =====================================================
    # COLLECT FRAME DATA
    # =====================================================

    def observe(
        self,
        frame_number,
        report,
        landmarks
    ):

        if not isinstance(report, dict):
            return

        try:
            frame = int(frame_number)
        except (
            TypeError,
            ValueError,
            OverflowError
        ):
            return

        if frame < 1 or frame <= self.last_frame:
            return

        self.last_frame = frame

        # Record first confirmed impact only.
        impact = report.get("impact")

        if (
            isinstance(impact, dict)
            and impact.get("event") is True
            and self.impact_frame is None
        ):

            try:
                event_frame = int(
                    impact.get("impact_frame", frame)
                )
            except (
                TypeError,
                ValueError,
                OverflowError
            ):
                event_frame = frame

            self.impact_frame = (
                event_frame
                if 1 <= event_frame <= frame
                else frame
            )

            self.contact_evidence = (
                self._contact_snapshot(impact)
            )

        # Only real, non-predicted ball samples.
        ball = report.get("ball")

        if (
            report.get("ball_detected")
            and isinstance(ball, dict)
            and ball.get("predicted") is False
            and ball.get("missing_frames", 0) == 0
        ):

            center = ball.get("center")

            if (
                isinstance(center, (tuple, list))
                and len(center) == 2
            ):

                try:
                    bx = float(center[0])
                    by = float(center[1])
                except (
                    TypeError,
                    ValueError,
                    OverflowError
                ):
                    pass
                else:
                    if (
                        math.isfinite(bx)
                        and math.isfinite(by)
                    ):
                        self.ball_samples.append({
                            "frame": frame,
                            "x": bx,
                            "y": by,
                        })

        # Prevent stale landmark reuse.
        if not report.get("landmarks_detected"):
            return

        # This tracker does NOT filter by batting phase.
        if self.arm is None:

            for side in ("left", "right"):

                wrist = self._point(
                    landmarks,
                    f"{side}_wrist"
                )

                shoulder = self._point(
                    landmarks,
                    f"{side}_shoulder"
                )

                if (
                    wrist is not None
                    and shoulder is not None
                ):
                    self.arm = side
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

        self.wrist_samples.append({
            "frame": frame,
            "x": wrist[0] - shoulder[0],
            "y": wrist[1] - shoulder[1],
        })

    # =====================================================
    # TEMPORAL MOTION CALCULATION
    # =====================================================

    @staticmethod
    def _measure(
        items,
        max_step,
        coordinate_label
    ):

        dx = 0.0
        dy = 0.0
        distance = 0.0
        pairs = 0
        rejected_jumps = 0

        for first, second in zip(
            items,
            items[1:]
        ):

            # Never bridge missing-frame gaps.
            if (
                second["frame"] - first["frame"]
                != 1
            ):
                continue

            sx = second["x"] - first["x"]
            sy = second["y"] - first["y"]

            step = math.hypot(sx, sy)

            if step > max_step:
                rejected_jumps += 1
                continue

            dx += abs(sx)
            dy += abs(sy)
            distance += step
            pairs += 1

        return {
            "sample_count": len(items),
            "adjacent_motion_pairs": pairs,
            "rejected_jumps": rejected_jumps,
            "horizontal_motion": round(dx, 4),
            "vertical_motion": round(dy, 4),
            "path_length": round(distance, 4),
            "units": coordinate_label,
        }

    # =====================================================
    # GENERATE FINAL DIAGNOSTICS
    # =====================================================

    def summarize(self):

        if self.impact_frame is None:
            return {
                "status": "No confirmed impact",
                "verified_shot_type": "Uncertain",
                "motion_candidate": "None",
                "impact_frame": None,
                "limitations": [
                    "No confirmed impact event observed."
                ],
            }

        impact = self.impact_frame

        pre = [
            s for s in self.wrist_samples
            if impact - self.pre_frames
            <= s["frame"] < impact
        ]

        post = [
            s for s in self.wrist_samples
            if impact < s["frame"]
            <= impact + self.post_frames
        ]

        before_ball = [
            s for s in self.ball_samples
            if impact - self.pre_frames
            <= s["frame"] < impact
        ]

        after_ball = [
            s for s in self.ball_samples
            if impact < s["frame"]
            <= impact + self.post_frames
        ]

        pre_measure = self._measure(
            pre,
            0.30,
            "normalized wrist-to-shoulder displacement"
        )

        post_measure = self._measure(
            post,
            0.30,
            "normalized wrist-to-shoulder displacement"
        )

        pre_ball_measure = self._measure(
            before_ball,
            150.0,
            "source-image pixels"
        )

        post_ball_measure = self._measure(
            after_ball,
            150.0,
            "source-image pixels"
        )

        limitations = []
        candidate = "None"

        most_recent = (
            pre[-1]["frame"]
            if pre
            else None
        )

        if (
            pre_measure["adjacent_motion_pairs"] < 3
            or len(pre) < 4
        ):
            limitations.append(
                "Insufficient adjacent pre-impact "
                "wrist measurements."
            )

        elif (
            most_recent is None
            or impact - most_recent > 3
        ):
            limitations.append(
                "No recent wrist observation within "
                "three frames of impact."
            )

        else:

            x = pre_measure["horizontal_motion"]
            y = pre_measure["vertical_motion"]

            if max(x, y) < 0.08:

                limitations.append(
                    "Pre-impact wrist movement is "
                    "too small to characterize."
                )

            elif x >= 1.6 * max(y, 0.001):

                candidate = (
                    "Horizontal-dominant pre-impact "
                    "wrist movement"
                )

            elif y >= 1.6 * max(x, 0.001):

                candidate = (
                    "Vertical-dominant pre-impact "
                    "wrist movement"
                )

            else:

                candidate = (
                    "Mixed pre-impact wrist movement"
                )

        if (
            pre_measure["sample_count"]
            - pre_measure["adjacent_motion_pairs"]
            > 1
        ):
            limitations.append(
                "Missing pre-impact pose frames "
                "limit motion continuity."
            )

        if post_measure["adjacent_motion_pairs"] < 3:
            limitations.append(
                "Limited post-impact wrist measurements."
            )

        ball_verified = (
            pre_ball_measure["adjacent_motion_pairs"] >= 2
            and post_ball_measure["adjacent_motion_pairs"] >= 2
        )

        ball_status = (
            "Pre- and post-impact ball displacement observed"
            if ball_verified
            else "Ball trajectory across impact not verified"
        )

        if not ball_verified:
            limitations.append(
                "Insufficient real ball detections "
                "on both sides of impact "
                "to assess trajectory."
            )

        return {
            "status": (
                "Experimental wrist-motion description"
                if candidate != "None"
                else "Insufficient wrist evidence"
            ),
            "verified_shot_type": "Uncertain",
            "motion_candidate": candidate,
            "impact_frame": impact,
            "tracked_arm": self.arm,
            "window_frames": {
                "before": self.pre_frames,
                "after": self.post_frames,
            },
            "wrist_evidence": {
                "before_impact": pre_measure,
                "after_impact": post_measure,
                "most_recent_pre_impact_frame": most_recent,
            },
            "ball_evidence": {
                "status": ball_status,
                "before_impact": pre_ball_measure,
                "after_impact": post_ball_measure,
            },
            "contact_evidence": self.contact_evidence,
            "limitations": limitations,
            "note": (
                "Read-only 2D motion diagnostics. "
                "Phases are not used to select frames. "
                "Bat bounding boxes do not measure bat angle; "
                "wrist motion and screen-space ball displacement "
                "cannot verify Drive, Cut, Pull or Defense. "
                "Does not modify shot, scores, impact, "
                "or completion."
            ),
        }
