
import math
from analysis.ideal_ranges import IDEAL_RANGES


class TechniqueAnalyzer:

    def __init__(self):
        pass

    # ========================================
    # HELPER: VALIDATE JOINT ANGLE
    # ========================================

    @staticmethod
    def _angle(angles, joint_name):

        try:
            value = float(angles[joint_name])

            if (
                math.isfinite(value)
                and 0.0 <= value <= 180.0
            ):
                return value

        except (
            KeyError,
            TypeError,
            ValueError,
            OverflowError
        ):
            pass

        return None

    # ========================================
    # MAIN TECHNIQUE ANALYSIS
    # ========================================

    def analyze(
        self,
        angles,
        batting_stance="Unknown",
        feedback_mode="legacy"
    ):

        if not isinstance(angles, dict) or not angles:
            return []
        
        # ========================================
        # EXPERIMENTAL FEEDBACK MODE
        # ========================================

        mode = str(
            feedback_mode or "legacy"
        ).strip().lower()

        if mode == "aligned":
            return self._analyze_aligned(
                angles,
                batting_stance
            )

        if mode != "legacy":
            raise ValueError(
                f"Unsupported feedback mode: {mode}"
            )


        mistakes = []

        # ====================================
        # 1. IDENTIFY FRONT/BACK SIDES
        # ====================================

        # Left-handed batting:
        # Right side = Front
        # Left side = Back
        #
        # Right-handed batting:
        # Left side = Front
        # Right side = Back
        #
        # Unknown preserves the old mapping
        # for backward compatibility.

        if batting_stance == "Left-handed":

            front_side = "right"
            back_side = "left"

        else:

            front_side = "left"
            back_side = "right"

        # ====================================
        # 2. FRONT ELBOW
        # ====================================

        front_elbow = self._angle(
            angles,
            f"{front_side}_elbow"
        )

        if (
            front_elbow is not None
            and front_elbow < 130
        ):

            mistakes.append(
                "Front elbow is too low."
            )

        # ====================================
        # 3. BACK ELBOW
        # ====================================

        back_elbow = self._angle(
            angles,
            f"{back_side}_elbow"
        )

        if (
            back_elbow is not None
            and back_elbow < 90
        ):

            mistakes.append(
                "Back elbow collapses early."
            )

        # ====================================
        # 4. FRONT KNEE
        # ====================================

        front_knee = self._angle(
            angles,
            f"{front_side}_knee"
        )

        if (
            front_knee is not None
            and front_knee > 170
        ):

            mistakes.append(
                "Front knee is too straight."
            )

        # ====================================
        # 5. BACK KNEE
        # ====================================

        back_knee = self._angle(
            angles,
            f"{back_side}_knee"
        )

        if (
            back_knee is not None
            and back_knee < 120
        ):

            mistakes.append(
                "Back knee bends too much."
            )

        # ====================================
        # 6. FRONT SHOULDER
        # ====================================

        front_shoulder = self._angle(
            angles,
            f"{front_side}_shoulder"
        )

        if (
            front_shoulder is not None
            and front_shoulder < 110
        ):

            mistakes.append(
                "Front shoulder is dropping."
            )

        # ====================================
        # 7. BACK SHOULDER
        # ====================================

        back_shoulder = self._angle(
            angles,
            f"{back_side}_shoulder"
        )

        if (
            back_shoulder is not None
            and back_shoulder < 110
        ):

            mistakes.append(
                "Back shoulder is collapsing."
            )

        # ====================================
        # 8. FINAL TECHNIQUE FEEDBACK
        # ====================================

        if not mistakes:

            mistakes.append(
                "Excellent batting technique."
            )

        return mistakes
    
    # ========================================
    # EXPERIMENTAL REFERENCE-ALIGNED FEEDBACK
    # ========================================

    def _analyze_aligned(
        self,
        angles,
        batting_stance="Unknown"
    ):
        """
        Experimental joint-angle feedback.

        Uses the same reference ranges as scoring.

        Shoulder = upper-arm-to-torso angle.
        Hip = torso-to-thigh angle.

        A 5-degree tolerance reduces warnings
        for very small range deviations.

        These are experimental reference ranges,
        not clinically validated technique rules.
        """

        feedback = []

        stance = str(
            batting_stance or "Right-handed"
        ).strip().lower()

        is_left_handed = stance in (
            "left-handed",
            "left handed"
        )

        front_prefix = (
            "right_"
            if is_left_handed
            else "left_"
        )

        joint_labels = {
            "elbow": "elbow angle",
            "knee": "knee angle",
            "shoulder": "upper-arm-to-torso angle",
            "hip": "torso-to-thigh angle"
        }

        tolerance = 5.0

        for joint in IDEAL_RANGES:

            angle = self._angle(
                angles,
                joint
            )

            if angle is None:
                continue

            # Match the stance-aware scoring
            # reference-joint mapping.
            reference_joint = joint

            if is_left_handed:

                if joint.startswith("left_"):
                    reference_joint = (
                        "right_" + joint[5:]
                    )

                elif joint.startswith("right_"):
                    reference_joint = (
                        "left_" + joint[6:]
                    )

            if reference_joint not in IDEAL_RANGES:
                continue

            low, high = IDEAL_RANGES[
                reference_joint
            ]

            # Joint is inside the reference range.
            if low <= angle <= high:
                continue

            # Calculate distance from range.
            if angle < low:
                deviation = low - angle
                direction = "below"
            else:
                deviation = angle - high
                direction = "above"

            # Ignore small deviations in feedback.
            # Scoring remains unchanged.
            if deviation < tolerance:
                continue

            side = (
                "Front"
                if joint.startswith(front_prefix)
                else "Back"
            )

            joint_type = joint.split("_", 1)[1]

            label = joint_labels.get(
                joint_type,
                joint_type + " angle"
            )

            feedback.append(
                f"{side} {label} "
                f"({angle:.1f}°) is {direction} "
                f"the experimental reference "
                f"range ({low}–{high}°)."
            )

        # An empty list means no significant
        # deviations were flagged in available
        # joint measurements.
        #
        # It does NOT imply perfect technique.
        return feedback

