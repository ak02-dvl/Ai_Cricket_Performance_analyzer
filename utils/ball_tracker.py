from collections import deque
import math
import cv2


class BallTracker:
    """
    Robust cricket ball tracker.

    Features:
        - Validates YOLO detections
        - Rejects sudden jumps
        - Uses movement prediction
        - Smooths real detections
        - Predicts only short detection gaps
        - Does not add predicted points to red trajectory
        - Clears tracking after too many missing frames
    """

    def __init__(
        self,
        max_points=50,
        min_confidence=0.10,
        max_jump=450,
        max_missing_frames=3,
        smoothing_alpha=0.55,
        velocity_alpha=0.45
    ):

        # =====================================================
        # CONFIGURATION
        # =====================================================

        self.max_points = int(max_points)
        self.min_confidence = float(min_confidence)
        self.max_jump = float(max_jump)
        self.max_missing_frames = int(max_missing_frames)

        self.smoothing_alpha = float(smoothing_alpha)
        self.velocity_alpha = float(velocity_alpha)

        # =====================================================
        # TRAJECTORY
        # =====================================================

        self.path = deque(
            maxlen=self.max_points
        )

        # =====================================================
        # TRACKING STATE
        # =====================================================

        self.current_position = None
        self.previous_position = None

        self.velocity = (
            0.0,
            0.0
        )

        self.missing_frames = 0
        self.tracking = False
        self.trajectory_active = False

        self.last_confidence = 0.0
        self.last_box = None
        self.tracking_frames = 0

        # =====================================================
        # DEBUG
        # =====================================================

        self.debug = True

    # =========================================================
    # DISTANCE
    # =========================================================

    def distance(self, p1, p2):

        if p1 is None or p2 is None:
            return float("inf")

        try:

            dx = float(p1[0]) - float(p2[0])
            dy = float(p1[1]) - float(p2[1])

            return math.sqrt(
                dx * dx + dy * dy
            )

        except Exception:

            return float("inf")

    # =========================================================
    # NORMALIZE POINT
    # =========================================================

    def normalize_point(self, point):

        if point is None:
            return None

        try:

            if len(point) < 2:
                return None

            x = float(point[0])
            y = float(point[1])

            if not math.isfinite(x):
                return None

            if not math.isfinite(y):
                return None

            return (
                x,
                y
            )

        except Exception:

            return None

    # =========================================================
    # GET BALL CENTER
    # =========================================================

    def get_ball_center(self, ball):

        if ball is None:
            return None

        if isinstance(ball, dict):

            center = ball.get("center")

            if center is not None:

                return self.normalize_point(
                    center
                )

            if (
                "x" in ball
                and
                "y" in ball
            ):

                return self.normalize_point(
                    (
                        ball["x"],
                        ball["y"]
                    )
                )

        if isinstance(
            ball,
            (
                tuple,
                list
            )
        ):

            return self.normalize_point(
                ball
            )

        return None

    # =========================================================
    # GET CONFIDENCE
    # =========================================================

    def get_confidence(self, ball):

        if ball is None:
            return 0.0

        if not isinstance(ball, dict):
            return 1.0

        try:

            confidence = float(
                ball.get(
                    "confidence",
                    0.0
                )
            )

        except Exception:

            confidence = 0.0

        if not math.isfinite(confidence):
            confidence = 0.0

        return max(
            0.0,
            min(
                confidence,
                1.0
            )
        )

    # =========================================================
    # PREDICT POSITION
    # =========================================================

    def predict_position(self):

        if self.current_position is None:
            return None

        return (
            self.current_position[0] + self.velocity[0],
            self.current_position[1] + self.velocity[1]
        )

    # =========================================================
    # UPDATE VELOCITY
    # =========================================================

    def update_velocity(self, new_position):

        if (
            new_position is None
            or
            self.current_position is None
        ):

            return

        dx = (
            new_position[0]
            -
            self.current_position[0]
        )

        dy = (
            new_position[1]
            -
            self.current_position[1]
        )

        alpha = self.velocity_alpha

        self.velocity = (

            alpha * dx
            +
            (1.0 - alpha) * self.velocity[0],

            alpha * dy
            +
            (1.0 - alpha) * self.velocity[1]

        )

    # =========================================================
    # SMOOTH POSITION
    # =========================================================

    def smooth_position(self, detected_position):

        if self.current_position is None:
            return detected_position

        alpha = self.smoothing_alpha

        x = (
            alpha * detected_position[0]
            +
            (1.0 - alpha) * self.current_position[0]
        )

        y = (
            alpha * detected_position[1]
            +
            (1.0 - alpha) * self.current_position[1]
        )

        return (
            x,
            y
        )

    # =========================================================
    # CREATE PREDICTED BOX
    # =========================================================

    def create_predicted_box(self, center):

        if center is None:
            return None

        if self.last_box is not None:

            try:

                x1, y1, x2, y2 = self.last_box

                width = max(
                    4,
                    x2 - x1
                )

                height = max(
                    4,
                    y2 - y1
                )

            except Exception:

                width = 10
                height = 10

        else:

            width = 10
            height = 10

        cx = center[0]
        cy = center[1]

        return (

            int(round(cx - width / 2)),
            int(round(cy - height / 2)),
            int(round(cx + width / 2)),
            int(round(cy + height / 2))

        )

    # =========================================================
    # CREATE BALL RESULT
    # =========================================================

    def create_ball(
        self,
        original_ball,
        center,
        predicted=False
    ):

        if center is None:
            return None

        if isinstance(
            original_ball,
            dict
        ):

            result = original_ball.copy()

        else:

            result = {}

        result["center"] = (
            float(center[0]),
            float(center[1])
        )

        result["confidence"] = float(
            self.last_confidence
        )

        # -----------------------------------------------------
        # BOX
        # -----------------------------------------------------

        if predicted:

            predicted_box = (
                self.create_predicted_box(
                    center
                )
            )

            if predicted_box is not None:
                result["box"] = predicted_box

        elif self.last_box is not None:

            result["box"] = self.last_box

        # -----------------------------------------------------
        # TRACKING INFORMATION
        # -----------------------------------------------------

        result["tracked"] = True
        result["predicted"] = bool(predicted)

        result["missing_frames"] = int(
            self.missing_frames
        )

        result["velocity"] = (
            float(self.velocity[0]),
            float(self.velocity[1])
        )

        result["speed"] = float(
            self.get_speed()
        )

        return result

    # =========================================================
    # ADD REAL TRAJECTORY POINT
    # =========================================================

    def add_path_point(self, position):

        if position is None:
            return

        point = (
            int(round(position[0])),
            int(round(position[1]))
        )

        # Do not add duplicate points
        if len(self.path) > 0:

            previous = self.path[-1]

            if point == previous:
                return

        self.path.append(point)

    # =========================================================
    # ACCEPT VALID DETECTION
    # =========================================================

    def accept_detection(
        self,
        ball,
        detected_position
    ):

        # =====================================================
        # FIRST DETECTION
        # =====================================================

        if self.current_position is None:

            self.current_position = (
                detected_position
            )

            self.previous_position = (
                detected_position
            )

            self.velocity = (
                0.0,
                0.0
            )

        else:

            # Update velocity using the raw detection
            self.update_velocity(
                detected_position
            )

            # Smooth only accepted real detections
            smoothed = (
                self.smooth_position(
                    detected_position
                )
            )

            self.previous_position = (
                self.current_position
            )

            self.current_position = (
                smoothed
            )

        # =====================================================
        # METADATA
        # =====================================================

        self.last_confidence = (
            self.get_confidence(
                ball
            )
        )

        if isinstance(
            ball,
            dict
        ):

            box = ball.get(
                "box"
            )

            if box is not None:

                try:

                    self.last_box = (
                        int(box[0]),
                        int(box[1]),
                        int(box[2]),
                        int(box[3])
                    )

                except Exception:

                    pass

        # =====================================================
        # STATE
        # =====================================================

        self.missing_frames = 0
        self.tracking = True
        self.trajectory_active = True
        self.tracking_frames += 1

        # =====================================================
        # IMPORTANT
        #
        # Only REAL detections are added to the red path.
        # =====================================================

        self.add_path_point(
            self.current_position
        )

        return self.create_ball(
            ball,
            self.current_position,
            predicted=False
        )

    # =========================================================
    # MOVEMENT VALIDATION
    # =========================================================
    def is_reasonable_movement(
        self,
        detected_position
    ):

        if detected_position is None:
            return False

        if self.current_position is None:
            return True

        actual_distance = self.distance(
            detected_position,
            self.current_position
        )

        # -----------------------------------------------------
        # Reject large sudden jumps
        # -----------------------------------------------------

        if actual_distance > self.max_jump:

            if self.debug:

                print(
                    "Rejected large jump:",
                    round(actual_distance, 2),
                    "pixels"
                )

            return False

        # -----------------------------------------------------
        # Compare with predicted position
        #
        # Only use prediction after at least 2
        # real tracking frames have been established.
        # -----------------------------------------------------

        if self.tracking_frames >= 2:

            predicted = (
                self.predict_position()
            )

            if predicted is not None:

                prediction_distance = (
                    self.distance(
                        detected_position,
                        predicted
                    )
                )

                prediction_limit = max(
                    45.0,
                    self.max_jump * 0.45
                )

                if prediction_distance > prediction_limit:

                    if self.debug:

                        print(
                            "Rejected prediction mismatch:",
                            round(
                                prediction_distance,
                                2
                            ),
                            "pixels"
                        )

                    return False

        return True

   

    # =========================================================
    # HANDLE MISSING DETECTION
    # =========================================================

    def handle_missing_detection(self):

        if self.current_position is None:
            return None

        self.missing_frames += 1

        # -----------------------------------------------------
        # Stop tracking after missing-frame limit
        # -----------------------------------------------------

        if (
            self.missing_frames
            >
            self.max_missing_frames
        ):

            if self.debug:

                print(
                    "Ball tracking cleared after",
                    self.missing_frames,
                    "missing frames"
                )

            self.clear()

            return None

        # -----------------------------------------------------
        # Temporary prediction
        # -----------------------------------------------------

        predicted = (
            self.predict_position()
        )

        if predicted is None:
            return None

        self.previous_position = (
            self.current_position
        )

        self.current_position = (
            predicted
        )

        # -----------------------------------------------------
        # Reduce velocity
        # -----------------------------------------------------

        self.velocity = (

            self.velocity[0] * 0.70,
            self.velocity[1] * 0.70

        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # Do not add predicted points to self.path.
        # Predicted points are displayed separately in orange.
        # -----------------------------------------------------

        return self.create_ball(
            None,
            self.current_position,
            predicted=True
        )

    # =========================================================
    # MAIN UPDATE
    # =========================================================

    def update(self, ball):

        # -----------------------------------------------------
        # No YOLO detection
        # -----------------------------------------------------

        if ball is None:

            return self.handle_missing_detection()

        # -----------------------------------------------------
        # Extract center
        # -----------------------------------------------------

        detected_position = (
            self.get_ball_center(
                ball
            )
        )

        if detected_position is None:

            return self.handle_missing_detection()

        # -----------------------------------------------------
        # Validate confidence
        # -----------------------------------------------------

        confidence = (
            self.get_confidence(
                ball
            )
        )

        if confidence < self.min_confidence:

            return self.handle_missing_detection()

        # -----------------------------------------------------
        # First detection
        # -----------------------------------------------------

        if self.current_position is None:

            return self.accept_detection(
                ball,
                detected_position
            )

        # -----------------------------------------------------
        # Validate movement
        # -----------------------------------------------------

        if not self.is_reasonable_movement(
            detected_position
        ):

            return self.handle_missing_detection()

        # -----------------------------------------------------
        # Accept valid real detection
        # -----------------------------------------------------

        return self.accept_detection(
            ball,
            detected_position
        )

    # =========================================================
    # DRAW TRAJECTORY
    # =========================================================

    def draw(self, frame):

        if frame is None:
            return frame

        if len(self.path) < 2:
            return frame

        points = list(
            self.path
        )

        # =====================================================
        # DRAW ONLY VALID REAL-DETECTION SEGMENTS
        # =====================================================

        for i in range(
            1,
            len(points)
        ):

            p1 = (
                int(points[i - 1][0]),
                int(points[i - 1][1])
            )

            p2 = (
                int(points[i][0]),
                int(points[i][1])
            )

            segment_distance = (
                self.distance(
                    p1,
                    p2
                )
            )

            # Never draw a large jump
            if segment_distance > self.max_jump:
                continue

            cv2.line(

                frame,

                p1,

                p2,

                (0, 0, 255),

                3,

                cv2.LINE_AA

            )

        # =====================================================
        # DRAW CURRENT BALL POSITION
        # =====================================================

        if self.current_position is not None:

            current = (

                int(
                    round(
                        self.current_position[0]
                    )
                ),

                int(
                    round(
                        self.current_position[1]
                    )
                )

            )

            # Orange = predicted position
            if self.missing_frames > 0:

                cv2.circle(

                    frame,

                    current,

                    8,

                    (0, 165, 255),

                    2

                )

                cv2.putText(

                    frame,

                    "PREDICTED",

                    (
                        current[0] + 10,
                        current[1]
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (0, 165, 255),

                    2,

                    cv2.LINE_AA

                )

            # Green = real detected position
            else:

                cv2.circle(

                    frame,

                    current,

                    7,

                    (0, 255, 0),

                    2

                )

        return frame

    # =========================================================
    # GET PATH
    # =========================================================

    def get_path(self):

        return list(
            self.path
        )

    # =========================================================
    # GET CURRENT POSITION
    # =========================================================

    def get_current_position(self):

        if self.current_position is None:
            return None

        return (

            float(
                self.current_position[0]
            ),

            float(
                self.current_position[1]
            )

        )

    # =========================================================
    # GET VELOCITY
    # =========================================================

    def get_velocity(self):

        return (

            float(
                self.velocity[0]
            ),

            float(
                self.velocity[1]
            )

        )

    # =========================================================
    # GET SPEED
    # =========================================================

    def get_speed(self):

        return math.sqrt(

            self.velocity[0] ** 2
            +
            self.velocity[1] ** 2

        )

    # =========================================================
    # IS TRACKING
    # =========================================================

    def is_tracking(self):

        return bool(
            self.tracking
        )

    # =========================================================
    # GET MISSING FRAMES
    # =========================================================

    def get_missing_frames(self):

        return int(
            self.missing_frames
        )

    # =========================================================
    # CLEAR TRACKER
    # =========================================================

    def clear(self):

        self.path.clear()

        self.current_position = None
        self.previous_position = None

        self.velocity = (
            0.0,
            0.0
        )

        self.missing_frames = 0
        self.tracking = False
        self.trajectory_active = False

        self.last_confidence = 0.0
        self.last_box = None
        self.tracking_frames = 0