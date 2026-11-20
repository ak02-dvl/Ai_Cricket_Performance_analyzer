
import math
from statistics import median


class SwingAnalyzer:

    ACTIVE_PHASES = {
        "backlift",
        "downswing",
        "impact",
        "follow through",
        "follow-through",
        "recovery"
    }

    MAX_FRAME_GAP = 5
    MIN_SUMMARY_SAMPLES = 3

    # ========================================
    # INITIALIZATION
    # ========================================

    def __init__(self):
        self.reset()

    # ========================================
    # RESET
    # ========================================

    def reset(self):
        self.prev_wrist = None
        self.wrist_side = None
        self.prev_frame_number = None
        self.prev_phase = None

        self.call_count = 0
        self.motion_samples = []
        self.motion_phases = {}
        self.sample_sides = set()
        # NEW: Store the phase and source frame
        # corresponding to each motion sample.
        self.motion_sample_phases = []
        self.motion_sample_frames = []

    # ========================================
    # VALIDATE WRIST LANDMARK
    # ========================================

    @staticmethod
    def _point(landmarks, name):

        if not isinstance(landmarks, dict):
            return None

        value = landmarks.get(name)

        if not isinstance(value, dict):
            return None

        try:
            x = float(value["x"])
            y = float(value["y"])

        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError
        ):
            return None

        if not (
            math.isfinite(x)
            and math.isfinite(y)
        ):
            return None

        return (x, y)

    # ========================================
    # PHASE TIMING LABEL
    # ========================================

    @staticmethod
    def _timing(phase):

        label = str(
            phase or ""
        ).strip().lower()

        if label in ("stance", "backlift"):
            return "Preparing"

        if label == "downswing":
            return "Swinging"

        if label == "impact":
            return "Impact Phase"

        if label in (
            "follow through",
            "follow-through",
            "recovery"
        ):
            return "Completed"

        return "Unknown"

    # ========================================
    # FRAME-BY-FRAME SWING ANALYSIS
    # ========================================

    def analyze(
        self,
        landmarks,
        phase,
        frame_number=None
    ):

        timing = self._timing(phase)

        phase_key = str(
            phase or ""
        ).strip().lower()

        self.call_count += 1

        # Use original video frame numbers when
        # available to account for skipped frames.

        if frame_number is None:
            current_frame = self.call_count

        elif (
            isinstance(frame_number, int)
            and not isinstance(frame_number, bool)
            and frame_number > 0
        ):
            current_frame = frame_number

        else:
            raise ValueError(
                "frame_number must be a positive integer"
            )

        # ====================================
        # SELECT WRIST
        # ====================================

        preferred = (
            (self.wrist_side,)
            if self.wrist_side
            else ()
        )

        order = preferred + tuple(
            side
            for side in ("left", "right")
            if side not in preferred
        )

        wrist = None
        side_used = None

        for side in order:

            candidate = self._point(
                landmarks,
                f"{side}_wrist"
            )

            if candidate is not None:
                wrist = candidate
                side_used = side
                break

        # ====================================
        # MISSING WRIST
        # ====================================

        if wrist is None:

            self.prev_wrist = None
            self.prev_frame_number = None
            self.prev_phase = None

            return {
                "speed": "Unknown",
                "direction": "Unknown",
                "timing": timing
            }

        # ====================================
        # WRIST SWITCH
        # ====================================

        if self.wrist_side != side_used:

            self.prev_wrist = None
            self.prev_frame_number = None
            self.prev_phase = None

        self.wrist_side = side_used

        speed = "Unknown"
        direction = "Unknown"

        # ====================================
        # CALCULATE WRIST MOVEMENT
        # ====================================

        if self.prev_wrist is not None:

            dx = (
                wrist[0] -
                self.prev_wrist[0]
            )

            dy = (
                wrist[1] -
                self.prev_wrist[1]
            )

            distance = math.hypot(
                dx,
                dy
            )

            # =================================
            # EXISTING SPEED LABELS
            # =================================

            if distance > 0.08:
                speed = "High"

            elif distance > 0.03:
                speed = "Medium"

            else:
                speed = "Low"

            # =================================
            # EXISTING DIRECTION LABELS
            # =================================

            if dx > 0.02:
                direction = "Screen Right"

            elif dx < -0.02:
                direction = "Screen Left"

            elif distance <= 0.02:
                direction = "Minimal Motion"

            else:
                direction = "Mostly Vertical"

            # =================================
            # NEW: TEMPORAL SWING TRACKING
            # =================================

            frame_gap = (
                current_frame -
                self.prev_frame_number
                if self.prev_frame_number is not None
                else None
            )

            if (
                frame_gap is not None
                and 1 <= frame_gap <= self.MAX_FRAME_GAP
                and self.prev_phase in self.ACTIVE_PHASES
                and phase_key in self.ACTIVE_PHASES
            ):

                movement_per_frame = (
                    distance / frame_gap
                )

                self.motion_samples.append(
                    movement_per_frame
                )
                # NEW: Keep metadata aligned with the
                # original accepted motion sample.
                self.motion_sample_phases.append(
                    phase_key
                )

                self.motion_sample_frames.append(
                    current_frame
                )

                self.motion_phases[phase_key] = (
                    self.motion_phases.get(
                        phase_key, 0
                    ) + 1
                )

                self.sample_sides.add(
                    side_used
                )

        # ====================================
        # SAVE CURRENT FRAME
        # ====================================

        self.prev_wrist = wrist
        self.prev_frame_number = current_frame
        self.prev_phase = phase_key

        # Keep the previous report format.
        return {
            "speed": speed,
            "direction": direction,
            "timing": timing
        }


    # ========================================
    # NEW: WRIST MOTION RELIABILITY
    # ========================================

    def _motion_reliability(self):

        samples = self.motion_samples
        phases = self.motion_sample_phases
        frames = self.motion_sample_frames

        candidate_indices = []

        # Compare each sample with its two
        # preceding and two following samples.
        # Skip phase boundaries.

        for i in range(2, len(samples) - 2):

            local_phases = phases[i - 2:i + 3]

            if len(set(local_phases)) != 1:
                continue

            # Avoid comparing across large
            # or invalid source-frame gaps.

            invalid_gap = any(
                not (
                    1 <= frames[j + 1] - frames[j]
                    <= self.MAX_FRAME_GAP
                )
                for j in range(i - 2, i + 2)
            )

            if invalid_gap:
                continue

            neighbors = [
                samples[i - 2],
                samples[i - 1],
                samples[i + 1],
                samples[i + 2]
            ]

            local_median = median(neighbors)

            deviations = [
                abs(value - local_median)
                for value in neighbors
            ]

            local_mad = median(deviations)

            # Conservative heuristic:
            # Require a sufficiently large
            # isolated jump before flagging it.

            threshold = max(
                0.08,
                3.0 * local_median,
                local_median
                + 6.0 * max(local_mad, 0.005)
            )

            immediate_neighbors = max(
                samples[i - 1],
                samples[i + 1]
            )

            if (
                samples[i] > threshold
                and samples[i] > 2.0 * immediate_neighbors
            ):
                candidate_indices.append(i)

        # Keep raw samples untouched.
        # Filtering is only for the new
        # diagnostic summary.

        excluded = set(candidate_indices)

        filtered = [
            value
            for index, value in enumerate(samples)
            if index not in excluded
        ]

        if filtered:

            filtered_average = round(
                sum(filtered) / len(filtered), 4
            )

            filtered_median = round(
                median(filtered), 4
            )

            # 90th percentile is a more
            # robust peak estimate than max().

            ordered = sorted(filtered)

            position = 0.90 * (len(ordered) - 1)

            lower = int(position)
            upper = min(
                lower + 1,
                len(ordered) - 1
            )

            weight = position - lower

            robust_peak = round(
                ordered[lower]
                + (
                    ordered[upper] - ordered[lower]
                ) * weight,
                4
            )

        else:

            filtered_average = None
            filtered_median = None
            robust_peak = None

        status = (
            "Evaluated"
            if len(samples) >= 5
            else "Insufficient samples"
        )

        return {
            "status": status,
            "raw_sample_count": len(samples),
            "candidate_spike_count": len(
                candidate_indices
            ),
            "candidate_source_frames": [
                frames[i]
                for i in candidate_indices
            ],
            "filtered_sample_count": len(filtered),
            "filtered_average_wrist_motion_per_frame":
                filtered_average,
            "filtered_median_wrist_motion_per_frame":
                filtered_median,
            "robust_peak_wrist_motion_per_frame":
                robust_peak,
            "note": (
                "Experimental phase-local spike "
                "detection. Flagged movement may "
                "represent genuine rapid motion. "
                "Filtering does not change original "
                "measurements or batting scores."
            )
        }
    
    # ========================================
    # NEW: PHASE-WISE SWING ANALYSIS
    # ========================================

    def _phase_wise_analysis(self, reliability):

        phase_order = (
            ("backlift", "Backlift"),
            ("downswing", "Downswing"),
            ("impact", "Impact"),
            ("follow through", "Follow Through"),
        )

        grouped = {
            key: [] for key, _ in phase_order
        }

        # ====================================
        # 1. GROUP MOTION SAMPLES BY PHASE
        # ====================================

        for movement, phase, frame in zip(
            self.motion_samples,
            self.motion_sample_phases,
            self.motion_sample_frames,
        ):

            phase_key = (
                str(phase)
                .strip()
                .lower()
                .replace("-", " ")
            )

            if phase_key == "recovery":
                phase_key = "follow through"

            if phase_key in grouped:
                grouped[phase_key].append(
                    (frame, movement)
                )

        # ====================================
        # 2. GET FLAGGED TRACKING SPIKES
        # ====================================

        spike_frames = set(
            reliability.get(
                "candidate_source_frames", []
            )
        )

        phases = {}

        # ====================================
        # 3. ANALYZE EACH BATTING PHASE
        # ====================================

        for phase_key, label in phase_order:

            observations = grouped[phase_key]

            kept = [
                value
                for frame, value in observations
                if frame not in spike_frames
            ]

            flagged = [
                frame
                for frame, _ in observations
                if frame in spike_frames
            ]

            count = len(observations)
            kept_count = len(kept)

            # ================================
            # SAMPLE RELIABILITY STATUS
            # ================================

            if kept_count >= 3:
                status = "Measured"

            elif kept_count:
                status = "Limited samples"

            else:
                status = "No qualifying samples"

            # ================================
            # ORIGINAL MAXIMUM MOVEMENT
            # ================================

            if observations:

                raw_peak_frame, raw_peak = max(
                    observations,
                    key=lambda item: item[1]
                )

                raw_peak = round(raw_peak, 4)

            else:

                raw_peak_frame = None
                raw_peak = None

            # ================================
            # FILTERED AVERAGE AND MEDIAN
            # ================================

            if kept:

                average = round(
                    sum(kept) / kept_count, 4
                )

                middle = round(
                    median(kept), 4
                )

            else:

                average = None
                middle = None

            # ================================
            # ROBUST PEAK (90TH PERCENTILE)
            # ================================

            if kept_count >= 3:

                ordered = sorted(kept)

                index = 0.90 * (kept_count - 1)

                lower = int(index)

                upper = min(
                    lower + 1,
                    kept_count - 1
                )

                weight = index - lower

                robust_peak = round(
                    ordered[lower]
                    + (
                        ordered[upper]
                        - ordered[lower]
                    ) * weight,
                    4
                )

            else:

                robust_peak = None

            # ================================
            # PHASE RESULT
            # ================================

            phases[phase_key] = {
                "label": label,
                "status": status,
                "raw_samples": count,
                "candidate_spikes": len(flagged),
                "candidate_spike_frames": flagged,
                "filtered_samples": kept_count,
                "raw_peak_wrist_motion_per_frame":
                    raw_peak,
                "raw_peak_source_frame":
                    raw_peak_frame,
                "filtered_average_wrist_motion_per_frame":
                    average,
                "filtered_median_wrist_motion_per_frame":
                    middle,
                "robust_peak_wrist_motion_per_frame":
                    robust_peak,
            }

        # ====================================
        # 4. FINAL PHASE-WISE REPORT
        # ====================================

        return {
            "phases": phases,
            "note": (
                "Experimental phase-grouped "
                "normalized wrist motion. "
                "No samples does not mean no "
                "physical movement. "
                "Small sample counts and tracking "
                "spikes limit reliability. "
                "Not physical bat speed or "
                "a coaching score."
            ),
        }
    
    # ========================================
    # NEW: IMPACT-ALIGNED SWING DIAGNOSTICS
    # ========================================

    def get_impact_alignment(
        self,
        confirmed_impact_frame,
        window=8
    ):

        # ====================================
        # 1. VALIDATE IMPACT REFERENCE
        # ====================================

        if (
            isinstance(confirmed_impact_frame, bool)
            or not isinstance(confirmed_impact_frame, int)
            or confirmed_impact_frame <= 0
        ):
            return {
                "status": "No confirmed impact frame available",
                "confirmed_impact_frame": None,
                "samples_near_impact": []
            }

        window = max(0, int(window))

        nearby = []

        # ====================================
        # 2. COLLECT QUALIFYING SWING SAMPLES
        # ====================================

        for frame, phase, movement in zip(
            self.motion_sample_frames,
            self.motion_sample_phases,
            self.motion_samples
        ):

            offset = (
                frame - confirmed_impact_frame
            )

            if abs(offset) > window:
                continue

            nearby.append({
                "source_frame": frame,
                "frames_from_impact": offset,
                "phase_passed_to_swing": phase,
                "wrist_motion_per_frame": round(
                    movement, 4
                )
            })

        # ====================================
        # 3. SAMPLE AT THE IMPACT EVENT FRAME
        # ====================================

        at_impact = next(
            (
                item
                for item in nearby
                if item["source_frame"]
                == confirmed_impact_frame
            ),
            None
        )

        phase_at_impact = (
            at_impact["phase_passed_to_swing"]
            if at_impact is not None
            else None
        )

        # ====================================
        # 4. LOCATE IMPACT-LABELED SAMPLE
        # ====================================

        impact_labeled = [
            item
            for item in nearby
            if str(
                item["phase_passed_to_swing"]
            ).strip().lower() == "impact"
        ]

        nearest_impact = (
            min(
                impact_labeled,
                key=lambda item: abs(
                    item["frames_from_impact"]
                )
            )
            if impact_labeled
            else None
        )

        # ====================================
        # 5. REPORT ALIGNMENT STATUS
        # ====================================

        if at_impact is None:

            status = (
                "No qualifying wrist-motion "
                "sample at impact frame"
            )

        elif str(
            phase_at_impact
        ).strip().lower() == "impact":

            status = (
                "Swing sample labeled Impact "
                "at confirmed frame"
            )

        else:

            status = (
                "Different swing phase label "
                "at confirmed impact frame"
            )

        # ====================================
        # 6. FINAL DIAGNOSTIC
        # ====================================

        return {
            "status": status,
            "confirmed_impact_frame":
                confirmed_impact_frame,
            "inspection_window_frames": window,
            "qualifying_samples_in_window":
                len(nearby),
            "phase_of_sample_at_impact":
                phase_at_impact,
            "nearest_impact_labeled_sample_frame": (
                nearest_impact["source_frame"]
                if nearest_impact is not None
                else None
            ),
            "impact_label_offset_frames": (
                nearest_impact["frames_from_impact"]
                if nearest_impact is not None
                else None
            ),
            "samples_near_impact": nearby,
            "note": (
                "Diagnostic only. Phase labels "
                "are those passed to SwingAnalyzer "
                "for qualifying motion samples. "
                "The confirmed impact event and "
                "existing phase logic are unchanged."
            )
        }



    # ========================================
    # NEW: FULL-SHOT SWING SUMMARY
    # ========================================

    def get_summary(self):

        samples = len(
            self.motion_samples
        )

        if samples >= self.MIN_SUMMARY_SAMPLES:
            status = "Measured"

        else:
            status = "Insufficient motion samples"

        if samples:

            peak_motion = round(
                max(self.motion_samples), 4
            )

            average_motion = round(
                sum(self.motion_samples) / samples,
                4
            )

            median_motion = round(
                median(self.motion_samples),
                4
            )

        else:

            peak_motion = None
            average_motion = None
            median_motion = None

        if len(self.sample_sides) == 1:

            tracked_wrist = next(
                iter(self.sample_sides)
            )

        elif self.sample_sides:

            tracked_wrist = "mixed"

        else:

            tracked_wrist = "Unknown"

        summary={
            "status": status,
            "samples": samples,
            "peak_wrist_motion_per_frame": peak_motion,
            "average_wrist_motion_per_frame": average_motion,
            "median_wrist_motion_per_frame": median_motion,
            "tracked_wrist": tracked_wrist,
            "samples_by_phase": dict(
                self.motion_phases
            ),
            "note": (
                "Experimental normalized wrist "
                "displacement per source frame. "
                "Not actual bat speed or a "
                "validated coaching score."
            )
        }
         # NEW: Add reliability diagnostics
        # without modifying existing values.

        
        # ====================================
        # MOTION RELIABILITY
        # ====================================

        reliability = self._motion_reliability()

        summary["motion_reliability"] = (
            reliability
        )

        # ====================================
        # PHASE-WISE SWING PERFORMANCE
        # ====================================

        summary["phase_analysis"] = (
            self._phase_wise_analysis(
                reliability
            )
        )

        return summary


