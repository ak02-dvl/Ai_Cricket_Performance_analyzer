"""Impact-window detector/tracker association audit (read-only).

Consumes the exported DetectionEvidenceV33 dictionary, not BallTracker state.
Flags disagreements for inspection; cannot identify the tracker's exact
rejection branch, prove a raw candidate is the ball, or recognize shot type.
"""

import math
from collections import Counter


class TrackerAssociationV34:
    def __init__(self, frame_width, frame_height,
                 disagreement_fraction=0.08,
                 raw_step_fraction=0.10):
        self.width = int(frame_width)
        self.height = int(frame_height)
        if self.width < 1 or self.height < 1:
            raise ValueError("Frame dimensions must be positive")
        diagonal = math.hypot(self.width, self.height)
        self.disagreement_limit_px = max(45.0, diagonal * float(disagreement_fraction))
        self.raw_step_limit_px = max(50.0, diagonal * float(raw_step_fraction))
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.disagreement_limit_px, self.raw_step_limit_px)):
            raise ValueError("Review thresholds must be positive and finite")

    @staticmethod
    def _point(value):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        try:
            point = [float(value[0]), float(value[1])]
        except (ValueError, TypeError, OverflowError):
            return None
        if not all(math.isfinite(v) for v in point):
            return None
        return point

    def _inside(self, point):
        return (0 <= point[0] < self.width and
                0 <= point[1] < self.height)

    def summarize(self, detection_audit):
        """Return a new JSON-safe report without modifying supplied data."""
        if not isinstance(detection_audit, dict):
            return self._unavailable("Detection audit v3.3 is unavailable")
        anchor = detection_audit.get("impact_frame")
        rows = detection_audit.get("frame_evidence")
        if not isinstance(anchor, int) or anchor < 1 or not isinstance(rows, list):
            return self._unavailable("No usable impact-window audit")

        issues = []
        observed_frames = set()
        by_region = {"before": Counter(), "impact": Counter(), "after": Counter()}
        previous_raw = None
        previous_raw_frame = None
        previous_frame = 0

        for row in rows:
            if not isinstance(row, dict):
                continue
            frame = row.get("frame")
            if not isinstance(frame, int) or isinstance(frame, bool):
                continue
            if frame <= previous_frame or frame < 1:
                continue
            previous_frame = frame
            observed_frames.add(frame)
            region = "before" if frame < anchor else "after" if frame > anchor else "impact"
            counts = by_region[region]
            counts["observed_frames"] += 1
            source = str(row.get("ball_source", "unknown"))
            counts["source_" + source] += 1
            raw = self._point(row.get("raw_ball_center"))
            tracked = self._point(row.get("tracked_ball_center"))

            if raw is not None:
                counts["valid_raw_centers"] += 1
                if not self._inside(raw):
                    counts["raw_out_of_frame"] += 1
                    issues.append({"frame": frame, "region": region,
                                   "kind": "raw_out_of_frame", "raw_center": raw})
                if previous_raw is not None and previous_raw_frame == frame - 1:
                    delta = math.dist(raw, previous_raw)
                    if delta > self.raw_step_limit_px:
                        counts["large_raw_step"] += 1
                        issues.append({"frame": frame, "region": region,
                                       "kind": "large_consecutive_raw_step",
                                       "previous_frame": previous_raw_frame,
                                       "distance_px": round(delta, 2)})
                # Do not connect observations across missing frames.
                previous_raw = raw
                previous_raw_frame = frame
            else:
                previous_raw = None
                previous_raw_frame = None

            if tracked is not None:
                counts["valid_tracked_centers"] += 1
                if not self._inside(tracked):
                    counts["tracked_out_of_frame"] += 1
                    issues.append({"frame": frame, "region": region,
                                   "kind": "tracked_out_of_frame",
                                   "tracked_center": tracked})

            if raw is not None and tracked is not None:
                gap = math.dist(raw, tracked)
                counts["raw_and_tracked"] += 1
                if gap > self.disagreement_limit_px:
                    counts["large_raw_tracker_gap"] += 1
                    issues.append({"frame": frame, "region": region,
                                   "kind": "raw_tracker_disagreement",
                                   "distance_px": round(gap, 2),
                                   "raw_center": raw,
                                   "tracked_center": tracked})
            elif raw is not None and tracked is None:
                counts["raw_without_tracker_position"] += 1
                issues.append({"frame": frame, "region": region,
                               "kind": "raw_without_tracker_position"})
            elif raw is None and tracked is not None:
                counts["tracker_without_raw_position"] += 1

        # Findings are review flags, never asserted reasons for rejection.
        after = by_region["after"]
        limitations = [
            "Raw/tracked separation does not identify which BallTracker.update() gate fired.",
            "A raw ball candidate can be a false positive or a different object.",
            "Thresholds flag screen-space anomalies; they are not validated physical speed limits.",
        ]
        if after["observed_frames"] == 0:
            limitations.append("No audited post-impact frames available.")
        if after["source_fresh_raw_ball"] == 0:
            limitations.append("No fresh raw ball candidate observed after impact.")
        if after["large_raw_tracker_gap"] > 0:
            limitations.append("Large post-impact raw/tracker separation requires video inspection.")
        if after["tracked_out_of_frame"] > 0:
            limitations.append("Tracked position extends outside image bounds after impact.")
        return {
            "status": "Review flags available" if issues else "No review flags observed",
            "impact_frame": anchor,
            "verified_shot_type": "Uncertain",
            "review_thresholds_px": {
                "raw_tracker_disagreement": round(self.disagreement_limit_px, 2),
                "consecutive_raw_step": round(self.raw_step_limit_px, 2),
            },
            "windows": {k: dict(sorted(v.items())) for k, v in by_region.items()},
            "issue_count": len(issues),
            "issues": issues,
            "limitations": limitations,
            "note": (
                "Read-only v3.3 provenance analysis, no tracker/detector mutation. "
                "No internal gate decision is claimed; this does not measure bat angle, "
                "3D velocity, impact truth, or the cricket shot type."
            ),
        }

    @staticmethod
    def _unavailable(reason):
        return {
            "status": "Unavailable", "impact_frame": None,
            "verified_shot_type": "Uncertain", "issue_count": 0,
            "issues": [], "limitations": [reason],
        }
