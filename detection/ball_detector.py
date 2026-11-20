from ultralytics import YOLO
import cv2


class BallDetector:

    def __init__(self):

        self.model = YOLO("yolov8n.pt")

    def detect_ball(self, frame):

        results = self.model(frame, verbose=False)

        best_ball = None
        best_conf = 0

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # COCO sports ball class
                if cls != 32:
                    continue

                if conf > best_conf:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    best_ball = (
                        x1,
                        y1,
                        x2,
                        y2,
                        conf
                    )

                    best_conf = conf

        return best_ball

    def draw_ball(self, frame, ball):

        if ball is None:
            return frame

        x1, y1, x2, y2, conf = ball

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        radius = max((x2 - x1), (y2 - y1)) // 2

        cv2.circle(
            frame,
            (cx, cy),
            radius,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Ball {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        return frame