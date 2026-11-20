
import math
from collections import deque
from statistics import median


class FootworkAnalyzer:

    FRONT_KNEE_STRAIGHT_ANGLE = 170.0

    MIN_MOTION_SAMPLES = 3
    MAX_FRAME_GAP = 5

    # Experimental threshold:
    # Leg-length fractions per video frame.
    
    MOTION_THRESHOLD = 0.06
    MIN_HIP_SHIFT_SAMPLES = 3
    HIP_SHIFT_THRESHOLD = 0.015

    # Front-foot movement detection
    MIN_FRONT_FOOT_SAMPLES = 3
    FRONT_FOOT_STEP_THRESHOLD = 0.08
    FRONT_FOOT_SMALL_STEP_THRESHOLD = 0.04

    def __init__(self):
        self.batting_stance = "Unknown"
        self.reset()

    def set_batting_stance(self, stance):

        allowed = (
            "Unknown",
            "Right-handed",
            "Left-handed"
        )

        if stance not in allowed:
            raise ValueError(
                "Invalid batting stance"
            )

        self.batting_stance = stance    

    # ========================================
    # RESET TEMPORAL TRACKING
    # ========================================

    def reset(self):
    # Back ankle tracking
        self._previous_back_ankle = None
        self._recent_back_ankle_motion = deque(
            maxlen=5
        )
        # Front-foot movement tracking
        self._front_ankle_history = deque(maxlen=40)
        self._front_ankle_baseline_y = None
        self._front_ankle_max_forward_ratio = 0.0
        self._front_ankle_last_frame = None
        self._front_ankle_scales = deque(maxlen=5)

        # NEW: Hip shift tracking
        self._previous_hip_projection = None
        self._recent_hip_shifts = deque(
            maxlen=5
        )
        # NEW: Screen-relative hip movement tracking
        self._previous_relative_hip_x = None
        self._recent_relative_hip_changes = deque(
            maxlen=5
        )

    # ========================================
    # DISTANCE
    # ========================================

    @staticmethod
    def distance(p1, p2):
        return math.hypot(
            p1["x"] - p2["x"],
            p1["y"] - p2["y"]
        )

    # ========================================
    # KNEE ANGLE
    # ========================================

    @staticmethod
    def calculate_angle(p1, p2, p3):

        try:
            ax = float(p1["x"]) - float(p2["x"])
            ay = float(p1["y"]) - float(p2["y"])

            bx = float(p3["x"]) - float(p2["x"])
            by = float(p3["y"]) - float(p2["y"])

            norm = (
                math.hypot(ax, ay) *
                math.hypot(bx, by)
            )

            if not math.isfinite(norm) or norm == 0:
                return None

            cosine = (ax * bx + ay * by) / norm

            if not math.isfinite(cosine):
                return None

            cosine = max(-1.0, min(1.0, cosine))

            return math.degrees(math.acos(cosine))

        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError
        ):
            return None

    # ========================================
    # CONVERT LANDMARK TO VIDEO PIXELS
    # ========================================

    @staticmethod
    def _pixel_point(landmarks, part, crop_box):

        try:
            x1, y1, x2, y2 = map(float, crop_box)

            point = landmarks[part]

            x = (
                x1 +
                float(point["x"]) * (x2 - x1)
            )

            y = (
                y1 +
                float(point["y"]) * (y2 - y1)
            )

            values = (x, y, x1, y1, x2, y2)

            if (
                all(math.isfinite(v) for v in values)
                and x2 > x1
                and y2 > y1
            ):
                return (x, y)

        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError
        ):
            pass

        return None

    # ========================================
    # TEMPORAL BACK-FOOT MOVEMENT
    # ========================================

    def _back_ankle_motion(
        self,
        landmarks,
        frame_number,
        crop_box,
        back_side="right"
    ):

        unknown = {
            "status": "Insufficient consecutive observations",
            "relative_motion_per_frame": None,
            "samples": 0
        }

        if (
            isinstance(frame_number, bool)
            or not isinstance(frame_number, int)
            or frame_number < 1
            or crop_box is None
        ):
            self.reset()

            return {
                **unknown,
                "status": "Frame coordinates unavailable"
            }

        ankle = self._pixel_point(
            landmarks,
            f"{back_side}_ankle",
            crop_box
        )

        hip = self._pixel_point(
            landmarks,
            f"{back_side}_hip",
            crop_box
        )

        if ankle is None or hip is None:

            self.reset()

            return {
                **unknown,
                "status": "Back ankle or hip unavailable"
            }

        leg_size = math.dist(ankle, hip)

        if (
            not math.isfinite(leg_size)
            or leg_size < 1.0
        ):

            self.reset()

            return {
                **unknown,
                "status": "Invalid body scale"
            }

        # ------------------------------------
        # COMPARE WITH PREVIOUS OBSERVATION
        # ------------------------------------

        if self._previous_back_ankle is not None:

            old_frame, old_point, old_scale = (
                self._previous_back_ankle
            )

            frame_gap = frame_number - old_frame

            if 1 <= frame_gap <= self.MAX_FRAME_GAP:

                displacement = math.dist(
                    ankle,
                    old_point
                )

                average_leg_size = max(
                    1.0,
                    (leg_size + old_scale) / 2
                )

                relative_motion = (
                    displacement /
                    (average_leg_size * frame_gap)
                )

                self._recent_back_ankle_motion.append(
                    relative_motion
                )

            else:
                # Prevent stale or widely separated
                # frames from being treated as continuous.
                self._recent_back_ankle_motion.clear()

        self._previous_back_ankle = (
            frame_number,
            ankle,
            leg_size
        )

        sample_count = len(
            self._recent_back_ankle_motion
        )

        # ------------------------------------
        # WAIT FOR SUFFICIENT EVIDENCE
        # ------------------------------------

        if sample_count < self.MIN_MOTION_SAMPLES:

            return {
                **unknown,
                "samples": sample_count
            }

        # Median reduces single-frame jitter.
        motion = median(
            self._recent_back_ankle_motion
        )

        if motion > self.MOTION_THRESHOLD:

            status = "Noticeable recent ankle movement"

        else:

            status = "Low recent ankle movement"

        return {
            "status": status,
            "relative_motion_per_frame": round(
                motion, 4
            ),
            "samples": sample_count
        }
    
    # ========================================
    # TEMPORAL FRONT-FOOT MOVEMENT
    # ========================================

    def _front_ankle_motion(
        self,
        landmarks,
        frame_number,
        crop_box,
        front_side="left"
    ):

        def unavailable(reason):
            self._front_ankle_history.clear()
            self._front_ankle_scales.clear()
            self._front_ankle_baseline_y = None
            self._front_ankle_max_forward_ratio = 0.0
            self._front_ankle_last_frame = None

            return {
                "status": reason,
                "forward_step_ratio": None,
                "samples": 0
            }

        if (
            isinstance(frame_number, bool)
            or not isinstance(frame_number, int)
            or frame_number < 1
            or crop_box is None
        ):
            return unavailable("Frame coordinates unavailable")

        ankle = self._pixel_point(
            landmarks,
            f"{front_side}_ankle",
            crop_box
        )

        hip = self._pixel_point(
            landmarks,
            f"{front_side}_hip",
            crop_box
        )

        if ankle is None or hip is None:
            return unavailable("Front ankle or hip unavailable")

        leg_size = math.dist(ankle, hip)

        if not math.isfinite(leg_size) or leg_size < 5:
            return unavailable("Invalid front-leg scale")

        previous_frame = self._front_ankle_last_frame

        if previous_frame is not None:
            frame_gap = frame_number - previous_frame

            if frame_gap <= 0:
                return unavailable("Invalid frame sequence")

            if frame_gap > 15:
                self._front_ankle_history.clear()
                self._front_ankle_scales.clear()
                self._front_ankle_baseline_y = None
                self._front_ankle_max_forward_ratio = 0.0

        self._front_ankle_last_frame = frame_number

        self._front_ankle_history.append(
            (frame_number, ankle[1])
        )

        self._front_ankle_scales.append(leg_size)

        # Establish baseline from the first three observations.
        if self._front_ankle_baseline_y is None:

            if len(self._front_ankle_history) < 3:
                return {
                    "status": "Establishing front-foot baseline",
                    "forward_step_ratio": None,
                    "samples": len(self._front_ankle_history)
                }

            initial_positions = [
                item[1]
                for item in list(self._front_ankle_history)[:3]
            ]

            self._front_ankle_baseline_y = median(
                initial_positions
            )

        baseline_y = self._front_ankle_baseline_y
        scale = median(self._front_ankle_scales)

        # Positive image-Y movement means downward movement.
        forward_ratio = (
            ankle[1] - baseline_y
        ) / scale

        if not math.isfinite(forward_ratio):
            return unavailable("Invalid front-foot measurement")

        self._front_ankle_max_forward_ratio = max(
            self._front_ankle_max_forward_ratio,
            forward_ratio
        )

        maximum = self._front_ankle_max_forward_ratio
        samples = len(self._front_ankle_history)

        if maximum >= self.FRONT_FOOT_STEP_THRESHOLD:
            status = "Forward step observed"

        elif maximum >= self.FRONT_FOOT_SMALL_STEP_THRESHOLD:
            status = "Small forward movement observed"

        else:
            status = "No clear forward step observed"

        return {
            "status": status,
            "forward_step_ratio": round(maximum, 4),
            "samples": samples
        }
    
    # ========================================
    # NEW: TEMPORAL HIP-SHIFT INDICATOR
    # ========================================

    def _hip_shift_indicator(
        self,
        landmarks,
        frame_number,
        crop_box
    ):

        def unavailable(reason):

            self._previous_hip_projection = None
            self._recent_hip_shifts.clear()

            return {
                "status": reason,
                "relative_shift_per_frame": None,
                "samples": 0
            }

        # ====================================
        # VALIDATE FRAME INFORMATION
        # ====================================

        if (
            isinstance(frame_number, bool)
            or not isinstance(frame_number, int)
            or frame_number < 1
            or crop_box is None
        ):
            return unavailable(
                "Frame coordinates unavailable"
            )

        # ====================================
        # GET LANDMARKS IN VIDEO COORDINATES
        # ====================================

        names = (
            "left_hip",
            "right_hip",
            "left_ankle",
            "right_ankle"
        )

        points = {
            name: self._pixel_point(
                landmarks,
                name,
                crop_box
            )
            for name in names
        }

        if any(
            point is None
            for point in points.values()
        ):
            return unavailable(
                "Hip or ankle landmarks unavailable"
            )

        front = points["left_ankle"]
        back = points["right_ankle"]

        left_hip = points["left_hip"]
        right_hip = points["right_hip"]

        # ====================================
        # CALCULATE FOOT AXIS
        # ====================================

        foot_dx = front[0] - back[0]
        foot_dy = front[1] - back[1]

        foot_spacing = math.hypot(
            foot_dx,
            foot_dy
        )

        leg_scale = (
            math.dist(left_hip, front) +
            math.dist(right_hip, back)
        ) / 2.0
        
        if (
            not math.isfinite(foot_spacing)
            or not math.isfinite(leg_scale)
            or foot_spacing < max(
                8.0,
                0.25 * leg_scale
            )
        ):

            result = unavailable(
                "Feet too close for reliable hip-shift estimate"
            )

            # ====================================
            # DIAGNOSTIC INFORMATION
            # ====================================

            valid_spacing = math.isfinite(
                foot_spacing
            )

            valid_scale = math.isfinite(
                leg_scale
            )

            required_spacing = (
                max(8.0, 0.25 * leg_scale)
                if valid_scale
                else None
            )

            separation_ratio = None

            if (
                valid_spacing
                and valid_scale
                and leg_scale > 0
            ):

                separation_ratio = round(
                    foot_spacing / leg_scale,
                    4
                )

            result["diagnostics"] = {
                "ankle_separation_px": (
                    round(foot_spacing, 2)
                    if valid_spacing
                    else None
                ),
                "leg_scale_px": (
                    round(leg_scale, 2)
                    if valid_scale
                    else None
                ),
                "required_separation_px": (
                    round(required_spacing, 2)
                    if required_spacing is not None
                    else None
                ),
                "separation_ratio": separation_ratio
            }

            return result



        # ====================================
        # CALCULATE HIP CENTER
        # ====================================

        hip_center = (
            (left_hip[0] + right_hip[0]) / 2.0,
            (left_hip[1] + right_hip[1]) / 2.0
        )

        # Project the hip center onto the
        # line from right ankle to left ankle.

        position = (
            (hip_center[0] - back[0]) * foot_dx
            +
            (hip_center[1] - back[1]) * foot_dy
        ) / (foot_spacing * foot_spacing)

        if (
            not math.isfinite(position)
            or abs(position) > 5
        ):
            return unavailable(
                "Invalid hip projection"
            )

        # ====================================
        # TEMPORAL HIP MOVEMENT
        # ====================================

        if self._previous_hip_projection is not None:

            old_frame, old_position = (
                self._previous_hip_projection
            )

            frame_gap = frame_number - old_frame

            if 1 <= frame_gap <= self.MAX_FRAME_GAP:

                change = (
                    position - old_position
                ) / frame_gap

                # Reject extreme jumps caused
                # by possible landmark errors.

                if abs(change) <= 0.20:

                    self._recent_hip_shifts.append(
                        change
                    )

                else:
                    self._recent_hip_shifts.clear()

            else:
                self._recent_hip_shifts.clear()

        # ====================================
        # SAVE CURRENT OBSERVATION
        # ====================================

        self._previous_hip_projection = (
            frame_number,
            position
        )

        samples = len(
            self._recent_hip_shifts
        )

        if samples < self.MIN_HIP_SHIFT_SAMPLES:

            return {
                "status": "Insufficient consecutive observations",
                "relative_shift_per_frame": None,
                "samples": samples
            }

        # ====================================
        # CLASSIFY RECENT HIP SHIFT
        # ====================================

        shift = median(
            self._recent_hip_shifts
        )

        if shift > self.HIP_SHIFT_THRESHOLD:

            status = (
                "Recent hip shift toward left ankle"
            )

        elif shift < -self.HIP_SHIFT_THRESHOLD:

            status = (
                "Recent hip shift toward right ankle"
            )

        else:

            status = (
                "No clear directional hip shift"
            )

        # ====================================
        # FINAL HIP SHIFT REPORT
        # ====================================

        return {
            "status": status,
            "relative_shift_per_frame": round(
                shift,
                4
            ),
            "samples": samples
        }


    # ==========================================
    # NEW: SCREEN-RELATIVE HIP MOVEMENT
    # ==========================================

    def _relative_hip_motion(
        self,
        landmarks,
        frame_number,
        crop_box
    ):

        def unavailable(reason):

            self._previous_relative_hip_x = None
            self._recent_relative_hip_changes.clear()

            return {
                "status": reason,
                "horizontal_shift_per_frame": None,
                "samples": 0
            }

        # ======================================
        # VALIDATE FRAME
        # ======================================

        if (
            isinstance(frame_number, bool)
            or not isinstance(frame_number, int)
            or frame_number < 1
            or crop_box is None
        ):
            return unavailable(
                "Frame coordinates unavailable"
            )

        # ======================================
        # EXTRACT BODY POSITIONS
        # ======================================

        try:

            names = (
                "left_hip",
                "right_hip",
                "left_ankle",
                "right_ankle"
            )

            points = {
                name: self._pixel_point(
                    landmarks,
                    name,
                    crop_box
                )
                for name in names
            }

            if any(
                point is None
                for point in points.values()
            ):
                return unavailable(
                    "Hip/ankle coordinates unavailable"
                )

            lh = points["left_hip"]
            rh = points["right_hip"]

            la = points["left_ankle"]
            ra = points["right_ankle"]

            # Average apparent leg length
            leg_scale = (
                math.dist(lh, la) +
                math.dist(rh, ra)
            ) / 2.0

            if (
                not math.isfinite(leg_scale)
                or leg_scale < 5.0
            ):
                return unavailable(
                    "Invalid body scale"
                )

            # ==================================
            # HIP CENTER RELATIVE TO FEET
            # ==================================

            hip_center_x = (
                lh[0] + rh[0]
            ) / 2.0

            ankle_center_x = (
                la[0] + ra[0]
            ) / 2.0

            relative_x = (
                hip_center_x - ankle_center_x
            ) / leg_scale

            if (
                not math.isfinite(relative_x)
                or abs(relative_x) > 5.0
            ):
                return unavailable(
                    "Invalid relative hip position"
                )

        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            ZeroDivisionError
        ):
            return unavailable(
                "Hip/ankle coordinates unavailable"
            )

        # ======================================
        # COMPARE WITH PREVIOUS FRAME
        # ======================================

        previous = self._previous_relative_hip_x

        if previous is not None:

            old_frame, old_relative_x = previous

            frame_gap = frame_number - old_frame

            if 1 <= frame_gap <= self.MAX_FRAME_GAP:

                change = (
                    relative_x - old_relative_x
                ) / frame_gap

                # Reject very large jumps that
                # may represent tracking errors.
                if (
                    math.isfinite(change)
                    and abs(change) <= 0.20
                ):

                    self._recent_relative_hip_changes.append(
                        change
                    )

                else:
                    self._recent_relative_hip_changes.clear()

            else:
                self._recent_relative_hip_changes.clear()

        # ======================================
        # SAVE CURRENT POSITION
        # ======================================

        self._previous_relative_hip_x = (
            frame_number,
            relative_x
        )

        samples = len(
            self._recent_relative_hip_changes
        )

        if samples < self.MIN_HIP_SHIFT_SAMPLES:

            return {
                "status": "Insufficient consecutive observations",
                "horizontal_shift_per_frame": None,
                "samples": samples
            }

        # ======================================
        # CALCULATE RECENT MOVEMENT
        # ======================================

        motion = median(
            self._recent_relative_hip_changes
        )

        return {
            "status": (
                "Measured (experimental "
                "screen-relative hip motion)"
            ),
            "horizontal_shift_per_frame": round(
                motion,
                4
            ),
            "samples": samples
        }

    
    # ========================================
    # BATTING STANCE IDENTIFICATION
    # ========================================

    def _stance_report(self, landmarks):

        # ====================================
        # 1. FRONT / BACK LEG LABELS
        # ====================================

        # Conventional right-handed batting:
        # left leg is front, right leg is back.
        #
        # Conventional left-handed batting:
        # right leg is front, left leg is back.
        #
        # These labels depend on the
        # user-selected batting stance.

        front_leg = {
            "Right-handed": "left",
            "Left-handed": "right"
        }.get(
            self.batting_stance,
            "Unknown"
        )

        back_leg = {
            "left": "right",
            "right": "left"
        }.get(
            front_leg,
            "Unknown"
        )

        report = {
            "batting_stance": self.batting_stance,

            "stance_source": (
                "User selection"
                if front_leg != "Unknown"
                else "Unspecified"
            ),

            "front_leg": front_leg,
            "back_leg": back_leg,

            "screen_left_leg": "Unknown",
            "screen_right_leg": "Unknown",

            "ankle_horizontal_separation": None,

            "geometry_status": (
                "Ankle positions unavailable"
            ),

            "note": (
                "Front/back labels assume a "
                "conventional batting stance. "
                "Existing technique scoring "
                "is not changed."
            )
        }

        # ====================================
        # 2. VALIDATE LANDMARKS
        # ====================================

        if not isinstance(landmarks, dict):
            return report

        try:

            left_x = float(
                landmarks["left_ankle"]["x"]
            )

            right_x = float(
                landmarks["right_ankle"]["x"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError
        ):
            return report

        if not (
            math.isfinite(left_x)
            and math.isfinite(right_x)
        ):
            return report

        # ====================================
        # 3. SCREEN-SPACE FOOT ORDER
        # ====================================

        separation = abs(
            left_x - right_x
        )

        report["ankle_horizontal_separation"] = (
            round(separation, 4)
        )

        if separation < 0.03:

            report["geometry_status"] = (
                "Ankles overlap horizontally "
                "in camera view"
            )

        else:

            if left_x < right_x:

                report["screen_left_leg"] = "left"
                report["screen_right_leg"] = "right"

            else:

                report["screen_left_leg"] = "right"
                report["screen_right_leg"] = "left"

            report["geometry_status"] = (
                "Screen left/right foot order observed"
            )

        return report


    # ========================================
    # MAIN FOOTWORK ANALYSIS
    # ========================================
    def analyze(
        self,
        landmarks,
        angles=None,
        frame_number=None,
        crop_box=None
    ):

        feedback = []
        footwork_score = 0

        required = [
            "nose",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle"
        ]

        if (
            not isinstance(landmarks, dict)
            or any(
                part not in landmarks
                for part in required
            )
        ):

            self.reset()

            return {
                "feedback": ["Insufficient landmarks"],
                "footwork_score": 0,
                "back_foot_motion": {
                    "status": "Insufficient landmarks",
                    "relative_motion_per_frame": None,
                    "samples": 0
                }
            }

        nose = landmarks["nose"]

        left_hip = landmarks["left_hip"]
        right_hip = landmarks["right_hip"]

        left_knee = landmarks["left_knee"]
        right_knee = landmarks["right_knee"]

        left_ankle = landmarks["left_ankle"]
        right_ankle = landmarks["right_ankle"]
        # ====================================
        # STANCE-AWARE FRONT / BACK JOINTS
        # ====================================

        front_side = (
            "right"
            if self.batting_stance == "Left-handed"
            else "left"
        )

        back_side = (
            "left"
            if front_side == "right"
            else "right"
        )

        front_hip = landmarks[f"{front_side}_hip"]
        front_knee = landmarks[f"{front_side}_knee"]
        front_ankle = landmarks[f"{front_side}_ankle"]

        back_knee = landmarks[f"{back_side}_knee"]
        back_ankle = landmarks[f"{back_side}_ankle"]

        # ====================================
        # 1. HEAD POSITION
        # ====================================

        hip_center_x = (
            left_hip["x"] +
            right_hip["x"]
        ) / 2

        if abs(nose["x"] - hip_center_x) < 0.06:

            feedback.append(
                "✓ Head position is stable"
            )

            footwork_score += 20

        else:

            feedback.append(
                "✗ Keep your head over your body"
            )

        # ====================================
        # 2. FRONT KNEE ANGLE
        # ====================================

        knee_angle = None

        if isinstance(angles, dict):

            try:
                value = float(
                    angles.get(f"{front_side}_knee")
                )

                if math.isfinite(value):
                    knee_angle = value

            except (
                TypeError,
                ValueError,
                OverflowError
            ):
                pass

        if knee_angle is None:

            knee_angle = self.calculate_angle(
                front_hip,
                front_knee,
                front_ankle
            )

        if knee_angle is None:

            feedback.append(
                "– Front knee bend could not be assessed"
            )

        elif knee_angle > self.FRONT_KNEE_STRAIGHT_ANGLE:

            feedback.append(
                "✗ Front knee is too straight"
            )

        else:

            feedback.append(
                "✓ Front knee is not excessively straight"
            )

            footwork_score += 20

        # ====================================
        # 3. FRONT-FOOT MOVEMENT
        # ====================================

        front_motion = self._front_ankle_motion(
            landmarks,
            frame_number,
            crop_box,
            front_side=front_side
        )

        feedback.append(
            "– Front-foot movement: " +
            front_motion["status"]
        )

        if front_motion["status"] == "Forward step observed":
            footwork_score += 40

        elif front_motion["status"] == "Small forward movement observed":
            footwork_score += 20

        # ====================================
        # 6. BACK-FOOT MOVEMENT
        # ====================================

        motion = self._back_ankle_motion(
            landmarks,
            frame_number,
            crop_box,
            back_side=back_side
        )

        feedback.append(
            "– Back-foot movement: " +
            motion["status"]
        )

        if motion["status"] in (
            "Noticeable recent ankle movement",
            "Low recent ankle movement"
        ):
            footwork_score += 20

        
        # ====================================
        # 7. NEW: TEMPORAL HIP SHIFT
        # ====================================

        hip_shift = self._hip_shift_indicator(
            landmarks,
            frame_number,
            crop_box
        )

        feedback.append(
            "– Hip-shift indicator: " +
            hip_shift["status"]
        )
        # ====================================
        # 8. FALLBACK: RELATIVE HIP MOTION
        # ====================================

        hip_motion = self._relative_hip_motion(
            landmarks,
            frame_number,
            crop_box
        )

        feedback.append(
            "– Hip-center movement: " +
            hip_motion["status"]
        )
        # ====================================
        # 9. BATTING STANCE IDENTIFICATION
        # ====================================

        stance = self._stance_report(
            landmarks
        )

        feedback.append(
            "– Batting stance: " +
            stance["batting_stance"]
        )





        # ====================================
        # FINAL RESULTS
        # ====================================
        return {
            "feedback": feedback,
            "footwork_score": footwork_score,
            "back_foot_motion": motion,
            "front_foot_motion": front_motion,
            "hip_shift": hip_shift,
            "hip_motion": hip_motion,
            "stance": stance
        }