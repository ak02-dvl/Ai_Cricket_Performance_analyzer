import cv2
import mediapipe as mp


class PoseDetector:

    def __init__(self):

        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_pose(self, frame):
        """
        Detect pose in the given frame.
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb)

        return results

    def draw_pose(self, frame, results):
        """
        Draw pose skeleton on the frame.
        """

        if results.pose_landmarks:

            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

        return frame

    def get_landmarks(self, results):
        """
        Return landmarks as normalized coordinates.
        """

        if not results.pose_landmarks:
            return None

        landmarks = {}

        MIN_VISIBILITY = 0.40

        for idx, landmark in enumerate(results.pose_landmarks.landmark):

            if landmark.visibility < MIN_VISIBILITY:
                continue

            name = self.mp_pose.PoseLandmark(idx).name.lower()

            landmarks[name] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            }

        return landmarks