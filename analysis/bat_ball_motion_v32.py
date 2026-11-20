
"""Read-only experimental bat and ball motion evidence."""

from collections import deque
import math


class BatBallMotionV32:

    def __init__(
        self,
        frame_width,
        frame_height,
        pre_frames=12,
        post_frames=10,
        min_bat_confidence=0.10,
        min_ball_confidence=0.15
    ):
        self.width = max(1, int(frame_width))
        self.height = max(1, int(frame_height))
        self.pre_frames = max(4, int(pre_frames))
        self.post_frames = max(4, int(post_frames))
        self.min_bat_confidence = float(min_bat_confidence)
        self.min_ball_confidence = float(min_ball_confidence)

        self.last_frame = 0
        self.impact_frame = None
        self.impact_geometry = None

        self.bat_samples = deque(maxlen=400)
        self.ball_samples = deque(maxlen=400)

    # =====================================================
    # VALIDATE NUMBERS AND DETECTION GEOMETRY
    # =====================================================

    @staticmethod
    def _num(value):
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError, OverflowError):
            return None

    @classmethod
    def _box(cls, data):
        if (
            not isinstance(data, (list, tuple))
            or len(data) != 4
        ):
            return None

        vals = [cls._num(v) for v in data]

        if any(v is None for v in vals):
            return None

        x1, y1, x2, y2 = vals

        return vals if x2 > x1 and y2 > y1 else None

    @classmethod
    def _center(cls, data):
        if (
            not isinstance(data, (list, tuple))
            or len(data) != 2
        ):
            return None

        x = cls._num(data[0])
        y = cls._num(data[1])

        if x is None or y is None:
            return None

        return x, y

    @staticmethod
    def _box_center(box):
        return (
            (box[0] + box[2]) / 2.0,
            (box[1] + box[3]) / 2.0
        )

    # =====================================================
    # SNAPSHOT AT CONFIRMED IMPACT
    # =====================================================

    def _impact_snapshot(self, impact):
        bat = impact.get("bat_input", {})
        ball = impact.get("ball_input", {})

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
                "status": "Missing accepted bat/ball boxes",
                "overlap_reported": bool(
                    impact.get("overlap", False)
                ),
                "bat_box_aspect_ratio": None,
            }

        bat_width = bat_box[2] - bat_box[0]
        bat_height = bat_box[3] - bat_box[1]

        bc = self._box_center(bat_box)
        ballc = self._box_center(ball_box)

        return {
            "status": "Accepted bat/ball boxes at impact",
            "overlap_reported": bool(
                impact.get("overlap", False)
            ),
            "bat_box_aspect_ratio": round(
                bat_height / bat_width, 3
            ),
            "box_center_distance_px": round(
                math.hypot(
                    bc[0] - ballc[0],
                    bc[1] - ballc[1]
                ),
                2
            ),
            "note": (
                "Bounding-box geometry is not the "
                "bat angle or physical contact point."
            ),
        }

    # =====================================================
    # COLLECT FRAME-BY-FRAME EVIDENCE
    # =====================================================

    def observe(self, frame_number, report, fresh_bat=None):

        if not isinstance(report, dict):
            return

        try:
            frame = int(frame_number)
        except (TypeError, ValueError, OverflowError):
            return

        if frame < 1 or frame <= self.last_frame:
            return

        self.last_frame = frame

        impact = report.get("impact")

        if (
            isinstance(impact, dict)
            and impact.get("event") is True
            and self.impact_frame is None
        ):
            try:
                anchor = int(
                    impact.get("impact_frame", frame)
                )
            except (TypeError, ValueError, OverflowError):
                anchor = frame

            self.impact_frame = (
                anchor if 1 <= anchor <= frame else frame
            )

            self.impact_geometry = (
                self._impact_snapshot(impact)
            )

        # Use only bat boxes identified as fresh.
        # The app supplies this from analyzer detection state.
        if (
            report.get("analysis_status") == "Analysis Working"
            and isinstance(fresh_bat, dict)
            and fresh_bat.get("fresh") is True
        ):
            conf = self._num(
                fresh_bat.get("confidence")
            )

            box = self._box(
                fresh_bat.get("box")
            )

            if (
                box is not None
                and conf is not None
                and self.min_bat_confidence <= conf <= 1.0
            ):
                x, y = self._box_center(box)

                self.bat_samples.append({
                    "frame": frame,
                    "x": x,
                    "y": y,
                    "confidence": conf,
                })

        # Reject predicted, stale or low-confidence
        # ball observations.
        ball = report.get("ball")

        if not (
            report.get("ball_detected")
            and isinstance(ball, dict)
        ):
            return

        if (
            ball.get("predicted") is not False
            or ball.get("is_predicted", False)
        ):
            return

        if ball.get("missing_frames", 0) != 0:
            return

        confidence = self._num(
            ball.get("confidence")
        )

        center = self._center(
            ball.get("center")
        )

        if (
            center is not None
            and confidence is not None
            and self.min_ball_confidence
            <= confidence <= 1.0
        ):
            self.ball_samples.append({
                "frame": frame,
                "x": center[0],
                "y": center[1],
                "confidence": confidence,
            })

    # =====================================================
    # SCREEN-SPACE MOVEMENT DIRECTION
    # =====================================================

    @staticmethod
    def _direction(dx, dy):

        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None

        horizontal = "right" if dx > 0 else "left"
        vertical = "down" if dy > 0 else "up"

        if abs(dx) > 2 * abs(dy):
            return "Screen " + horizontal

        if abs(dy) > 2 * abs(dx):
            return "Screen " + vertical

        return "Screen " + vertical + "-" + horizontal

    # =====================================================
    # MEASURE MOTION IN A SOURCE-FRAME WINDOW
    # =====================================================

    def _segment(
        self,
        observations,
        first,
        last,
        jump_limit,
        min_pairs
    ):
        samples = [
            s for s in observations
            if first <= s["frame"] <= last
        ]

        dx = 0.0
        dy = 0.0
        path = 0.0
        pairs = 0
        rejected = 0

        for a, b in zip(samples, samples[1:]):

            # Never calculate movement across missing frames.
            if b["frame"] - a["frame"] != 1:
                continue

            vx = b["x"] - a["x"]
            vy = b["y"] - a["y"]

            length = math.hypot(vx, vy)

            if length > jump_limit:
                rejected += 1
                continue

            dx += vx
            dy += vy
            path += length
            pairs += 1

        net = math.hypot(dx, dy)

        coherence = (
            net / path if path > 1e-9 else 0.0
        )

        coverage = (
            pairs / max(1, last - first)
        )

        reliable = (
            pairs >= min_pairs
            and coverage >= 0.25
            and net >= 2.0
            and coherence >= 0.60
        )

        return {
            "samples": len(samples),
            "adjacent_pairs": pairs,
            "rejected_jumps": rejected,
            "frame_coverage_ratio": round(
                coverage, 3
            ),
            "net_dx_px": round(dx, 2),
            "net_dy_px": round(dy, 2),
            "path_px": round(path, 2),
            "directional_consistency": round(
                coherence, 3
            ),
            "screen_direction": (
                self._direction(dx, dy)
                if reliable else None
            ),
            "status": (
                "Measured" if reliable
                else "Limited evidence"
            ),
            "first_sample_frame": (
                samples[0]["frame"]
                if samples else None
            ),
            "last_sample_frame": (
                samples[-1]["frame"]
                if samples else None
            ),
            "_dx": dx,
            "_dy": dy,
            "_reliable": reliable,
        }

    @staticmethod
    def _clean(segment):
        return {
            k: v for k, v in segment.items()
            if not k.startswith("_")
        }

    # =====================================================
    # FINAL ANALYSIS
    # =====================================================

    def summarize(self):

        if self.impact_frame is None:
            return {
                "status": "No confirmed impact",
                "verified_shot_type": "Uncertain",
                "impact_frame": None,
                "limitations": [
                    "No confirmed impact to anchor motion windows."
                ],
            }

        anchor = self.impact_frame

        jump_limit = max(
            40.0,
            0.25 * max(self.width, self.height)
        )

        bpre = self._segment(
            self.bat_samples,
            anchor - self.pre_frames,
            anchor - 1,
            jump_limit,
            2
        )

        bpost = self._segment(
            self.bat_samples,
            anchor + 1,
            anchor + self.post_frames,
            jump_limit,
            2
        )

        ppre = self._segment(
            self.ball_samples,
            anchor - self.pre_frames,
            anchor - 1,
            jump_limit,
            2
        )

        ppost = self._segment(
            self.ball_samples,
            anchor + 1,
            anchor + self.post_frames,
            jump_limit,
            2
        )

        pre_recent = (
            ppre["last_sample_frame"] is not None
            and anchor - ppre["last_sample_frame"] <= 3
        )

        post_recent = (
            ppost["first_sample_frame"] is not None
            and ppost["first_sample_frame"] - anchor <= 3
        )

        trajectory_ready = (
            ppre["_reliable"]
            and ppost["_reliable"]
            and pre_recent
            and post_recent
        )

        direction_change = None

        if trajectory_ready:

            v1 = (ppre["_dx"], ppre["_dy"])
            v2 = (ppost["_dx"], ppost["_dy"])

            cosine = (
                v1[0] * v2[0] + v1[1] * v2[1]
            ) / (
                math.hypot(*v1) * math.hypot(*v2)
            )

            direction_change = round(
                math.degrees(
                    math.acos(
                        max(-1.0, min(1.0, cosine))
                    )
                ),
                1
            )

        limitations = []

        if not bpre["_reliable"]:
            limitations.append(
                "Insufficient coherent fresh bat-box "
                "motion before impact."
            )

        if not bpost["_reliable"]:
            limitations.append(
                "Insufficient coherent fresh bat-box "
                "motion after impact."
            )

        if not ppre["_reliable"]:
            limitations.append(
                "Insufficient real ball-motion evidence "
                "before impact."
            )

        if not ppost["_reliable"]:
            limitations.append(
                "Insufficient real ball-motion evidence "
                "after impact."
            )

        if (
            ppre["_reliable"]
            and ppost["_reliable"]
            and not (pre_recent and post_recent)
        ):
            limitations.append(
                "Ball observations are too far from impact "
                "for trajectory comparison."
            )

        return {
            "status": (
                "Trajectory comparison available"
                if trajectory_ready
                else "Partial motion evidence"
            ),
            "verified_shot_type": "Uncertain",
            "impact_frame": anchor,
            "window_frames": {
                "before": self.pre_frames,
                "after": self.post_frames,
            },
            "bat_box_motion": {
                "before_impact": self._clean(bpre),
                "after_impact": self._clean(bpost),
                "source": (
                    "Fresh bat detections only; "
                    "tracked-only boxes excluded."
                ),
            },
            "ball_motion": {
                "before_impact": self._clean(ppre),
                "after_impact": self._clean(ppost),
                "trajectory_status": (
                    "Comparable near impact"
                    if trajectory_ready
                    else "Not verified across impact"
                ),
                "screen_direction_change_degrees": (
                    direction_change
                ),
                "note": (
                    "Screen-space ball movement only. "
                    "Not physical speed or 3D shot direction."
                ),
            },
            "impact_geometry": self.impact_geometry,
            "limitations": limitations,
            "note": (
                "Read-only experimental evidence. "
                "Bounding boxes do not measure bat angle "
                "or bat-tip position. Ball tracking may "
                "contain identity errors. No scoring, phase, "
                "impact or shot-completion logic is modified."
            ),
        }
