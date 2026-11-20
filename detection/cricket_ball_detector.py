from ultralytics import YOLO
import cv2
import math


class CricketBallDetector:
    """
    Cricket ball detector.

    IMPORTANT:
        This class performs DETECTION only.

        It does NOT predict the ball when YOLO misses it.

        Temporary tracking/prediction is handled by BallTracker.
    """

    def __init__(
        self,
        model_path="models/cricket_ball.pt",
        confidence_threshold=0.15,
        iou_threshold=0.45,
        max_detections=20,
        max_ball_area_ratio=0.02
    ):

        self.model = YOLO(model_path)

        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.max_ball_area_ratio = float(max_ball_area_ratio)

        self.debug = False

        print("\nCricket Ball Detector")
        print("-" * 50)
        print(f"Model: {model_path}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"IOU threshold: {self.iou_threshold}")
        print(f"Maximum detections: {self.max_detections}")
        print("-" * 50)

    # =========================================================
    # SAFE FLOAT
    # =========================================================

    def safe_float(self, value, default=0.0):

        try:
            value = float(value)

            if math.isfinite(value):
                return value

        except Exception:
            pass

        return default

    # =========================================================
    # DISTANCE
    # =========================================================

    def distance(self, p1, p2):

        if p1 is None or p2 is None:
            return float("inf")

        try:
            dx = float(p1[0]) - float(p2[0])
            dy = float(p1[1]) - float(p2[1])

            return math.sqrt(dx * dx + dy * dy)

        except Exception:
            return float("inf")

    # =========================================================
    # VALID BOX
    # =========================================================

    def valid_box(self, box, frame_width, frame_height):

        if box is None:
            return None

        try:
            if len(box) < 4:
                return None

            x1 = int(round(float(box[0])))
            y1 = int(round(float(box[1])))
            x2 = int(round(float(box[2])))
            y2 = int(round(float(box[3])))

        except Exception:
            return None

        if x2 < x1:
            x1, x2 = x2, x1

        if y2 < y1:
            y1, y2 = y2, y1

        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))

        if x2 <= x1:
            return None

        if y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    # =========================================================
    # CENTER
    # =========================================================

    def get_center(self, box):

        if box is None:
            return None

        try:
            x1, y1, x2, y2 = box

            return (
                int(round((x1 + x2) / 2.0)),
                int(round((y1 + y2) / 2.0))
            )

        except Exception:
            return None

    # =========================================================
    # BALL SIZE VALIDATION
    # =========================================================

    def valid_ball_size(
        self,
        box,
        frame_width,
        frame_height
    ):

        if box is None:
            return False

        try:
            x1, y1, x2, y2 = box

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                return False

            area = width * height

            frame_area = frame_width * frame_height

            if frame_area <= 0:
                return False

            area_ratio = area / frame_area

            # Reject very large objects.
            if area_ratio > self.max_ball_area_ratio:
                return False

            # Reject extremely wide/tall objects.
            aspect_ratio = (
                max(width, height) /
                max(1, min(width, height))
            )

            if aspect_ratio > 5.0:
                return False

            return True

        except Exception:
            return False

    # =========================================================
    # DETECT BALL
    # =========================================================

    def detect_ball(self, frame):

        if frame is None:
            return None

        if not hasattr(frame, "shape"):
            return None

        try:
            frame_height, frame_width = frame.shape[:2]

        except Exception:
            return None

        if frame_width <= 0 or frame_height <= 0:
            return None

        # =====================================================
        # YOLO
        # =====================================================

        try:

            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )

        except Exception as e:

            if self.debug:
                print(f"Ball YOLO error: {e}")

            return None

        if not results:
            return None

        candidates = []

        # =====================================================
        # COLLECT CANDIDATES
        # =====================================================

        for result in results:

            if result is None:
                continue

            if result.boxes is None:
                continue

            for detection in result.boxes:

                try:

                    confidence = float(
                        detection.conf[0].item()
                    )

                    if confidence < self.confidence_threshold:
                        continue

                    coordinates = (
                        detection.xyxy[0].tolist()
                    )

                    box = self.valid_box(
                        coordinates,
                        frame_width,
                        frame_height
                    )

                    if box is None:
                        continue

                    if not self.valid_ball_size(
                        box,
                        frame_width,
                        frame_height
                    ):
                        continue

                    center = self.get_center(box)

                    if center is None:
                        continue

                    x1, y1, x2, y2 = box

                    width = x2 - x1
                    height = y2 - y1

                    area = width * height

                    candidates.append({
                        "box": box,
                        "center": center,
                        "confidence": confidence,
                        "area": area
                    })

                except Exception as e:

                    if self.debug:
                        print(
                            f"Ball candidate error: {e}"
                        )

                    continue

        # =====================================================
        # NO VALID DETECTION
        # =====================================================

        if not candidates:
            return None

        # =====================================================
        # SELECT BEST DETECTION
        # =====================================================

        selected = max(
            candidates,
            key=lambda candidate:
            candidate["confidence"]
        )

        if self.debug:
            print(
                "Ball detected:",
                selected
            )

        return {
            "box": selected["box"],
            "center": selected["center"],
            "confidence": float(
                selected["confidence"]
            ),
            "area": selected["area"],
            "predicted": False,
            "tracked": False
        }

    # =========================================================
    # DRAW BALL
    # =========================================================

    def draw_ball(self, frame, ball):

        if frame is None:
            return frame

        if ball is None:
            return frame

        try:

            center = ball.get("center")

            if center is None:
                return frame

            cx = int(round(float(center[0])))
            cy = int(round(float(center[1])))

            frame_height, frame_width = frame.shape[:2]

            cx = max(
                0,
                min(cx, frame_width - 1)
            )

            cy = max(
                0,
                min(cy, frame_height - 1)
            )

            confidence = self.safe_float(
                ball.get("confidence", 0.0)
            )

            predicted = bool(
                ball.get("predicted", False)
            )

            tracked = bool(
                ball.get("tracked", False)
            )

            # =================================================
            # PREDICTED BALL
            # =================================================

            if predicted:

                cv2.circle(
                    frame,
                    (cx, cy),
                    8,
                    (0, 165, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "BALL PREDICTED",
                    (cx + 12, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA
                )

                return frame

            # =================================================
            # REAL DETECTED BALL
            # =================================================

            box = ball.get("box")

            if box is not None:

                try:

                    x1, y1, x2, y2 = box

                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2
                    )

                except Exception:
                    pass

            cv2.circle(
                frame,
                (cx, cy),
                7,
                (255, 0, 0),
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                3,
                (0, 255, 255),
                -1
            )

            label = f"BALL {confidence:.2f}"

            if tracked:
                label += " TRACKED"

            cv2.putText(
                frame,
                label,
                (
                    cx + 10,
                    max(20, cy - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

        except Exception as e:

            if self.debug:
                print(
                    f"Ball drawing warning: {e}"
                )

        return frame

    # =========================================================
    # CLEAR
    # =========================================================

    def clear(self):
        pass