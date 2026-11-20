from ultralytics import YOLO
import cv2


class ObjectDetector:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect_batsman(self, frame):

        results = self.model(frame, verbose=False)

        h, w = frame.shape[:2]

        # Ignore players on the left half of the frame
        frame_center = w // 2

        biggest_area = 0
        batsman_box = None

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])

                # Person class only
                if cls != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_x = (x1 + x2) // 2

                # Ignore left side of frame (usually bowler)
                if center_x < frame_center:
                    continue

                area = (x2 - x1) * (y2 - y1)

                if area > biggest_area:
                    biggest_area = area
                    batsman_box = (x1, y1, x2, y2)

        return batsman_box

    def draw_batsman(self, frame, box):

        if box is None:
            return frame

        x1, y1, x2, y2 = box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            "Tracked Batsman",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        return frame

    def crop_batsman(self, frame, box, padding=40):

        if box is None:
            return None, None

        x1, y1, x2, y2 = box

        h, w = frame.shape[:2]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        crop = frame[y1:y2, x1:x2]

        return crop, (x1, y1, x2, y2)