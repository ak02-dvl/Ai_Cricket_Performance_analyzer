"""Opt-in, read-only audit of actual detector-stage outputs.

Does not change YOLO decisions, the ball tracker, impact, phase, or scoring.
A detector-stage snapshot must be supplied by CricketAnalyzer; final
tracked-ball reports alone cannot establish raw detection provenance.
"""

from collections import Counter, deque
import math


class DetectionEvidenceV33:
    def __init__(self, pre_frames=12, post_frames=10,
                 min_bat_confidence=0.10, min_ball_confidence=0.15):
        self.pre_frames = max(1, int(pre_frames))
        self.post_frames = max(1, int(post_frames))
        self.min_bat_confidence = float(min_bat_confidence)
        self.min_ball_confidence = float(min_ball_confidence)
        self.impact_frame = None
        self.last_frame = 0
        self.frames = deque(maxlen=1500)

    @staticmethod
    def _number(value):
        try:
            v = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return v if math.isfinite(v) else None

    @classmethod
    def _box(cls, candidate):
        if not isinstance(candidate, dict):
            return None
        box = candidate.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        coords = [cls._number(v) for v in box]
        if any(v is None for v in coords):
            return None
        x1, y1, x2, y2 = coords
        return coords if x2 > x1 and y2 > y1 else None

    @classmethod
    def _center(cls, candidate):
        if not isinstance(candidate, dict):
            return None
        center = candidate.get("center")
        if isinstance(center, (tuple, list)) and len(center) == 2:
            vals = [cls._number(v) for v in center]
            if all(v is not None for v in vals):
                return vals
        box = cls._box(candidate)
        if box is not None:
            return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        return None

    def _bat_status(self, candidate):
        if candidate is None:
            return "no_selected_bat"
        if not isinstance(candidate, dict):
            return "invalid_bat_candidate"
        if candidate.get("tracker_only") is True:
            return "csrt_tracker_only"
        conf = self._number(candidate.get("confidence"))
        if self._box(candidate) is None or conf is None:
            return "invalid_bat_candidate"
        if conf < self.min_bat_confidence:
            return "selected_bat_below_audit_threshold"
        return "selected_bat_not_csrt"

    def _ball_status(self, raw, tracked):
        if isinstance(raw, dict):
            conf = self._number(raw.get("confidence"))
            if raw.get("predicted") or raw.get("is_predicted"):
                return "raw_marked_predicted"
            if conf is None or self._center(raw) is None:
                return "invalid_raw_ball"
            if conf < self.min_ball_confidence:
                return "raw_below_audit_threshold"
            return "fresh_raw_ball"

        if raw is not None:
            return "invalid_raw_ball"
        if isinstance(tracked, dict):
            if tracked.get("predicted") or tracked.get("is_predicted"):
                return "prediction_only_no_raw_ball"
            return "tracked_result_without_raw_ball"
        return "no_raw_or_tracked_ball"

    def observe(self, frame_number, report, detector_snapshot):
        """Read a single frame AFTER analyzer.process_frame returns."""
        try:
            frame = int(frame_number)
        except (TypeError, ValueError, OverflowError):
            return
        if frame <= self.last_frame or frame < 1:
            return
        self.last_frame = frame
        report = report if isinstance(report, dict) else {}

        impact = report.get("impact")
        if (self.impact_frame is None and isinstance(impact, dict)
                and impact.get("event") is True):
            try:
                anchor = int(impact.get("impact_frame", frame))
            except (TypeError, ValueError, OverflowError):
                anchor = frame
            self.impact_frame = anchor if 1 <= anchor <= frame else frame

        snapshot_ok = (
            isinstance(detector_snapshot, dict)
            and detector_snapshot.get("frame") == frame
            and detector_snapshot.get("stage") == "after_ball_tracker"
        )
        bat = detector_snapshot.get("bat_candidate") if snapshot_ok else None
        raw = detector_snapshot.get("raw_ball") if snapshot_ok else None
        tracked = detector_snapshot.get("tracked_ball") if snapshot_ok else None
        bat_status = self._bat_status(bat) if snapshot_ok else "stage_not_reached"
        ball_status = self._ball_status(raw, tracked) if snapshot_ok else "stage_not_reached"

        self.frames.append({
            "frame": frame,
            "analysis_status": str(report.get("analysis_status", "Unknown")),
            "bat_source": bat_status,
            "ball_source": ball_status,
            "bat_confidence": (
                self._number(bat.get("confidence")) if isinstance(bat, dict) else None
            ),
            "raw_ball_confidence": (
                self._number(raw.get("confidence")) if isinstance(raw, dict) else None
            ),
            "bat_center": self._center(bat) if snapshot_ok else None,
            "raw_ball_center": self._center(raw) if snapshot_ok else None,
            "tracked_ball_center": self._center(tracked) if snapshot_ok else None,
        })

    @staticmethod
    def _summarize_window(rows):
        bat_counts = Counter(x["bat_source"] for x in rows)
        ball_counts = Counter(x["ball_source"] for x in rows)
        static_pairs = 0
        previous = None
        for row in rows:
            if (previous is not None
                    and row["frame"] == previous["frame"] + 1
                    and row["bat_source"] == "selected_bat_not_csrt"
                    and previous["bat_source"] == "selected_bat_not_csrt"
                    and row["bat_center"] is not None
                    and previous["bat_center"] is not None
                    and math.dist(row["bat_center"], previous["bat_center"]) < 0.5):
                static_pairs += 1
            previous = row
        return {
            "observed_frames": len(rows),
            "bat_source_counts": dict(sorted(bat_counts.items())),
            "ball_source_counts": dict(sorted(ball_counts.items())),
            "selected_bat_candidate_frames": bat_counts["selected_bat_not_csrt"],
            "csrt_only_frames": bat_counts["csrt_tracker_only"],
            "fresh_raw_ball_frames": ball_counts["fresh_raw_ball"],
            "prediction_only_ball_frames": ball_counts["prediction_only_no_raw_ball"],
            "tracked_without_raw_frames": ball_counts["tracked_result_without_raw_ball"],
            "missing_raw_and_tracked_ball_frames": ball_counts["no_raw_or_tracked_ball"],
            "stage_not_reached_frames": ball_counts["stage_not_reached"],
            "repeated_selected_bat_center_pairs": static_pairs,
        }

    def summarize(self):
        if self.impact_frame is None:
            return {
                "status": "No confirmed impact",
                "impact_frame": None,
                "verified_shot_type": "Uncertain",
                "limitations": ["No impact event to anchor the detection audit."],
            }

        anchor = self.impact_frame
        pre = [x for x in self.frames if anchor - self.pre_frames <= x["frame"] < anchor]
        post = [x for x in self.frames if anchor < x["frame"] <= anchor + self.post_frames]
        at_impact = next((x for x in self.frames if x["frame"] == anchor), None)
        before = self._summarize_window(pre)
        after = self._summarize_window(post)
        issues = []
        if after["observed_frames"] < self.post_frames:
            issues.append("Post-impact inspection window was not fully processed.")
        if after["fresh_raw_ball_frames"] == 0:
            issues.append("No qualifying raw ball detections after impact in the audited window.")
        if after["prediction_only_ball_frames"]:
            issues.append("Some post-impact ball positions were tracker predictions, not fresh detections.")
        if after["tracked_without_raw_frames"]:
            issues.append("Some tracked ball results had no corresponding raw detector output.")
        if after["stage_not_reached_frames"]:
            issues.append("Some post-impact frames did not reach the detector snapshot stage.")
        if after["repeated_selected_bat_center_pairs"]:
            issues.append("Selected non-CSRT bat boxes repeated the same center; verify on video before calling this stale or stationary.")
        if before["fresh_raw_ball_frames"] == 0:
            issues.append("No qualifying raw ball detections before impact in the audited window.")

        return {
            "status": "Detection provenance audited",
            "impact_frame": anchor,
            "verified_shot_type": "Uncertain",
            "window_frames": {"before": self.pre_frames, "after": self.post_frames},
            "before_impact": before,
            "impact_frame_evidence": at_impact,
            "after_impact": after,
            "frame_evidence": pre + ([at_impact] if at_impact else []) + post,
            "limitations": issues,
            "note": (
                "Read-only provenance audit of accepted bat candidates, raw dedicated-ball detector "
                "output, and ball tracker output. A missing raw output does not establish why "
                "YOLO missed the ball (occlusion, model threshold, or absence are not distinguished). "
                "A repeated bat-box center does not prove a stale detection. An unflagged bat candidate is not necessarily a fresh YOLO prediction when upstream detectors have their own persistence. "
                "No impact, phase, scoring, shot-completion, or detector decision is modified."
            ),
        }
