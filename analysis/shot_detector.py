import math


class ShotDetector:

    def __init__(
        self,
        approach_distance=450,
        start_distance=350,
        max_distance=700,
        confirmation_frames=2
    ):

        # =====================================================
        # SETTINGS
        # =====================================================

        self.approach_distance = float(
            approach_distance
        )

        self.start_distance = float(
            start_distance
        )

        self.max_distance = float(
            max_distance
        )

        self.confirmation_frames = max(
            1,
            int(confirmation_frames)
        )

        # =====================================================
        # STATE
        # =====================================================

        self.previous_ball = None

        self.previous_distance = None

        self.approach_frames = 0

        self.shot_active = False

        self.frames_since_start = 0

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.previous_ball = None

        self.previous_distance = None

        self.approach_frames = 0

        self.shot_active = False

        self.frames_since_start = 0

    # =========================================================
    # CENTER OF BOX
    # =========================================================

    def get_box_center(
        self,
        box
    ):

        if box is None:
            return None

        try:

            x1, y1, x2, y2 = box

            return (
                (float(x1) + float(x2)) / 2.0,
                (float(y1) + float(y2)) / 2.0
            )

        except (
            TypeError,
            ValueError,
            IndexError
        ):

            return None

    # =========================================================
    # BALL CENTER
    # =========================================================

    def get_ball_center(
        self,
        ball
    ):

        if ball is None:
            return None

        # -----------------------------------------------------
        # DICTIONARY FORMAT
        # -----------------------------------------------------

        if isinstance(ball, dict):

            center = ball.get(
                "center"
            )

            if center is not None:

                try:

                    return (
                        float(center[0]),
                        float(center[1])
                    )

                except (
                    TypeError,
                    ValueError,
                    IndexError
                ):

                    return None

            # -------------------------------------------------
            # X / Y FORMAT
            # -------------------------------------------------

            if (
                "x" in ball
                and
                "y" in ball
            ):

                try:

                    return (
                        float(ball["x"]),
                        float(ball["y"])
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    return None

        # -----------------------------------------------------
        # DIRECT POINT
        # -----------------------------------------------------

        try:

            if len(ball) >= 2:

                return (
                    float(ball[0]),
                    float(ball[1])
                )

        except (
            TypeError,
            ValueError,
            IndexError
        ):

            pass

        return None

    # =========================================================
    # DISTANCE
    # =========================================================

    def calculate_distance(
        self,
        p1,
        p2
    ):

        if p1 is None or p2 is None:
            return None

        dx = p1[0] - p2[0]

        dy = p1[1] - p2[1]

        return math.sqrt(
            dx * dx +
            dy * dy
        )

    # =========================================================
    # BALL SPEED
    # =========================================================

    def calculate_ball_speed(
        self,
        current_ball
    ):

        if (
            current_ball is None
            or
            self.previous_ball is None
        ):

            return 0.0

        return self.calculate_distance(
            current_ball,
            self.previous_ball
        )

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        ball,
        batsman_box
    ):

        # =====================================================
        # GET POSITIONS
        # =====================================================

        ball_center = self.get_ball_center(
            ball
        )

        batsman_center = self.get_box_center(
            batsman_box
        )

        # =====================================================
        # NO BATSMAN
        # =====================================================

        if batsman_center is None:

            self.previous_ball = ball_center

            self.previous_distance = None

            self.approach_frames = 0

            return False

        # =====================================================
        # NO BALL
        # =====================================================

        if ball_center is None:

            self.previous_ball = None

            self.previous_distance = None

            self.approach_frames = 0

            return False

        # =====================================================
        # DISTANCE
        # =====================================================

        distance = self.calculate_distance(
            ball_center,
            batsman_center
        )

        if distance is None:

            return False

        # =====================================================
        # BALL SPEED
        # =====================================================

        ball_speed = self.calculate_ball_speed(
            ball_center
        )

        # =====================================================
        # FIRST BALL FRAME
        # =====================================================

        if self.previous_distance is None:

            self.previous_distance = distance

            self.previous_ball = ball_center

            return False

        # =====================================================
        # BALL APPROACHING
        # =====================================================

        approaching = (

            distance
            <
            self.previous_distance

        )

        # =====================================================
        # APPROACH CONFIRMATION
        # =====================================================

        if (

            approaching

            and

            distance
            <=
            self.approach_distance

        ):

            self.approach_frames += 1

        else:

            self.approach_frames = max(
                0,
                self.approach_frames - 1
            )

        # =====================================================
        # START SHOT
        # =====================================================

        if (

            not self.shot_active

            and

            distance
            <=
            self.start_distance

            and

            self.approach_frames
            >=
            self.confirmation_frames

        ):

            self.shot_active = True

            self.frames_since_start = 0

        # =====================================================
        # ACTIVE SHOT
        # =====================================================

        if self.shot_active:

            self.frames_since_start += 1

            # -------------------------------------------------
            # Keep shot active while ball is reasonably close
            # -------------------------------------------------

            if distance <= self.max_distance:

                result = True

            else:

                result = False

                self.reset()

                return result

        else:

            result = False

        # =====================================================
        # SAVE HISTORY
        # =====================================================

        self.previous_distance = distance

        self.previous_ball = ball_center

        # =====================================================
        # RETURN
        # =====================================================

        return result

    # =========================================================
    # STATUS
    # =========================================================

    def is_active(self):

        return self.shot_active