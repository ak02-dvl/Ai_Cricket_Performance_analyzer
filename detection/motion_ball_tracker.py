import cv2


class MotionBallTracker:

    def __init__(self):

        self.previous_gray = None

    def detect_ball(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self.previous_gray is None:

            self.previous_gray = gray

            return None

        diff = cv2.absdiff(self.previous_gray, gray)

        self.previous_gray = gray

        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best_ball = None

        smallest_area = float("inf")

        for cnt in contours:

            area = cv2.contourArea(cnt)

            # Ignore tiny noise
            if area < 10:
                continue

            # Ignore large moving objects
            if area > 250:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Ball should be almost circular
            ratio = w / float(h)

            if ratio < 0.6 or ratio > 1.4:
                continue

            if area < smallest_area:

                smallest_area = area

                best_ball = (x, y, w, h)

        return best_ball

    def draw_ball(self, frame, ball):

        if ball is None:
            return frame

        x, y, w, h = ball

        cx = x + w // 2
        cy = y + h // 2

        cv2.circle(
            frame,
            (cx, cy),
            max(w, h) // 2,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            "Ball",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        return frame