import math


class ShotClassifier:

    def __init__(self):

        pass

    # =====================================================
    # Get landmark
    # =====================================================

    def get_point(self, landmarks, name):

        if landmarks is None:
            return None

        point = landmarks.get(name)

        if point is None:
            return None

        if isinstance(point, dict):

            if "x" in point and "y" in point:

                return (
                    float(point["x"]),
                    float(point["y"])
                )

        elif isinstance(point, (tuple, list)):

            if len(point) >= 2:

                return (
                    float(point[0]),
                    float(point[1])
                )

        return None

    # =====================================================
    # Distance
    # =====================================================

    def distance(self, p1, p2):

        if p1 is None or p2 is None:
            return None

        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    # =====================================================
    # Bat angle
    # =====================================================

    def calculate_bat_angle(self, landmarks):

        left_wrist = self.get_point(
            landmarks,
            "left_wrist"
        )

        right_wrist = self.get_point(
            landmarks,
            "right_wrist"
        )

        left_elbow = self.get_point(
            landmarks,
            "left_elbow"
        )

        right_elbow = self.get_point(
            landmarks,
            "right_elbow"
        )

        if (
            left_wrist is None
            or right_wrist is None
        ):
            return None

        wrist_center = (

            (left_wrist[0] + right_wrist[0]) / 2,

            (left_wrist[1] + right_wrist[1]) / 2
        )

        if (
            left_elbow is None
            or right_elbow is None
        ):
            return None

        elbow_center = (

            (left_elbow[0] + right_elbow[0]) / 2,

            (left_elbow[1] + right_elbow[1]) / 2
        )

        dx = (
            wrist_center[0]
            - elbow_center[0]
        )

        dy = (
            wrist_center[1]
            - elbow_center[1]
        )

        angle = math.degrees(
            math.atan2(
                dy,
                dx
            )
        )

        angle = abs(angle)

        if angle > 180:
            angle = 360 - angle

        return angle

    # =====================================================
    # Main classifier
    # =====================================================

    def classify(
        self,
        angles,
        phase,
        landmarks=None,
        impact=None
    ):

        # =================================================
        # If landmarks aren't available, fallback
        # =================================================

        if landmarks is None:

            return self.basic_classification(
                angles,
                phase
            )

        # =================================================
        # Calculate bat angle
        # =================================================

        bat_angle = self.calculate_bat_angle(
            landmarks
        )

        # =================================================
        # Extract useful joint angles
        # =================================================

        left_knee = angles.get(
            "left_knee"
        )

        right_knee = angles.get(
            "right_knee"
        )

        left_elbow = angles.get(
            "left_elbow"
        )

        right_elbow = angles.get(
            "right_elbow"
        )

        left_shoulder = angles.get(
            "left_shoulder"
        )

        right_shoulder = angles.get(
            "right_shoulder"
        )

        # =================================================
        # DEFENSIVE SHOT
        #
        # This must be checked BEFORE pull shot.
        # =================================================

        defensive_score = 0

        # Bat relatively vertical
        if bat_angle is not None:

            vertical_difference = min(
                abs(bat_angle - 90),
                abs(bat_angle - 270)
            )

            if vertical_difference < 30:

                defensive_score += 3

        # Low elbow movement / controlled arms
        if left_elbow is not None:

            if 80 <= left_elbow <= 150:

                defensive_score += 1

        if right_elbow is not None:

            if 80 <= right_elbow <= 150:

                defensive_score += 1

        # Knee bend
        if left_knee is not None:

            if 130 <= left_knee <= 175:

                defensive_score += 1

        # Impact strongly supports defensive shot
        if impact is not None:

            if impact.get("detected", False):

                if impact.get(
                    "wrist_speed",
                    999
                ) < 120:

                    defensive_score += 2

        # Phase
        if phase in (
            "Downswing",
            "Impact"
        ):

            defensive_score += 1

        # =================================================
        # Strong defensive decision
        # =================================================

        if defensive_score >= 4:

            return "Defensive Shot"

        # =================================================
        # PULL SHOT
        #
        # Require actual evidence.
        # =================================================

        pull_score = 0

        if bat_angle is not None:

            # Horizontal-ish bat
            if (
                bat_angle < 45
                or
                bat_angle > 135
            ):

                pull_score += 2

        # Bent back knee can support pull
        if right_knee is not None:

            if 100 <= right_knee <= 150:

                pull_score += 1

        # Shoulder rotation
        if (
            left_shoulder is not None
            and
            right_shoulder is not None
        ):

            shoulder_difference = abs(
                left_shoulder -
                right_shoulder
            )

            if shoulder_difference > 20:

                pull_score += 1

        # Need impact or follow-through
        if phase == "Follow Through":

            pull_score += 2

        # =================================================
        # Only call Pull Shot with enough evidence
        # =================================================

        if pull_score >= 4:

            return "Pull Shot"

        # =================================================
        # Drive
        # =================================================

        drive_score = 0

        if left_knee is not None:

            if left_knee < 150:

                drive_score += 1

        if bat_angle is not None:

            if (
                45 <= bat_angle <= 135
            ):

                drive_score += 1

        if phase == "Follow Through":

            drive_score += 1

        if drive_score >= 3:

            return "Drive"

        # =================================================
        # Default
        # =================================================

        return "Controlled Shot"

    # =====================================================
    # Fallback
    # =====================================================

    def basic_classification(
        self,
        angles,
        phase
    ):

        if phase == "Follow Through":

            return "Shot Played"

        if phase == "Downswing":

            return "Controlled Shot"

        return "Preparing"