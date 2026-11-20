import cv2
import math
import os
from collections import deque
from copy import deepcopy

# =========================================================
# DETECTORS
# =========================================================
from detection.object_detector import ObjectDetector
from detection.cricket_ball_detector import CricketBallDetector
from detection.cricket_object_detector import CricketObjectDetector
from pose_estimation.pose_detector import PoseDetector

# =========================================================
# ANALYSIS MODULES
# =========================================================
from analysis.shot_detector import ShotDetector
from analysis.shot_classifier import ShotClassifier
from analysis.technique_analyzer import TechniqueAnalyzer
from analysis.swing_analyzer import SwingAnalyzer
from analysis.footwork_analyzer import FootworkAnalyzer
from analysis.batting_analysis import calculate_joint_angles
from analysis.scoring import calculate_score, calculate_technique_score
from analysis.phase_detector import PhaseDetector
from analysis.impact_detector import ImpactDetector
from analysis.bat_ball_contact import BatBallContactAnalyzer

# =========================================================
# UTILITIES
# =========================================================
from utils.smoothing import AngleSmoother
from utils.ball_tracker import BallTracker


class CricketAnalyzer:
    """
    =========================================================
                        AI CRICKET PERFORMANCE ANALYZER
    =========================================================
    Main processing pipeline.
    """

    def __init__(self):
        print()
        print("=" * 60)
        print("        INITIALIZING CRICKET ANALYZER")
        print("=" * 60)
        
        # =====================================================
        # EXPERIMENTAL JOINT-ANGLE GEOMETRY MODE
        # =====================================================

        self.angle_geometry_mode = os.getenv(
            "CRICKET_ANGLE_MODE",
            "legacy"
        ).strip().lower()

        if self.angle_geometry_mode not in ("legacy", "pixel"):
            raise ValueError(
                "CRICKET_ANGLE_MODE must be legacy or pixel"
            )

        print(
            f"[ANGLE MODE] {self.angle_geometry_mode}"
        )
        
        # =============================================
        # EXPERIMENTAL TECHNIQUE FEEDBACK MODE
        # =============================================

        self.technique_feedback_mode = os.getenv(
            "CRICKET_FEEDBACK_MODE",
            "legacy"
        ).strip().lower()

        if self.technique_feedback_mode not in (
            "legacy",
            "aligned"
        ):
            raise ValueError(
                "CRICKET_FEEDBACK_MODE must be "
                "'legacy' or 'aligned'"
            )

        print(
            "[TECHNIQUE FEEDBACK MODE] "
            f"{self.technique_feedback_mode}"
        )
        
        # =============================================
        # JOINT MEASUREMENT COVERAGE
        # =============================================

        self.joint_coverage_enabled = (
            os.getenv(
                "CRICKET_JOINT_COVERAGE",
                "0"
            ).strip() == "1"
        )




        # Person / Batsman Detector
        self.detector = ObjectDetector()

        # Cricket Object Detector (YOLO)
        self.cricket_object_detector = CricketObjectDetector(
            model_path="runs/detect/train-3/weights/best.pt",
            confidence=0.05,
            bat_confidence=0.10,
            batsman_confidence=0.15,
            ball_confidence=0.05
        )
        self.cricket_detector = self.cricket_object_detector

        # Dedicated Ball Detector
        self.ball_detector = CricketBallDetector(
            model_path="models/cricket_ball.pt",
            confidence_threshold=0.15
        )

        # Pose Detector
        self.pose = PoseDetector()

        # Ball Tracker
        self.ball_tracker = BallTracker(
            max_points=50,
            min_confidence=0.10,
            max_jump=500,
            max_missing_frames=6,
            smoothing_alpha=0.75,
            velocity_alpha=0.70
        )

        # Bat Tracker State
        self.bat_tracker = None
        self.bat_tracker_active = False
        self.last_valid_bat_box = None
        self.last_valid_bat_confidence = 0.0
        self.frames_since_last_bat = 0
        self.max_bat_missing_frames = 8

        # Angle Smoother
        self.smoother = AngleSmoother(window_size=7)

        # Analysis Modules
        self.shot_detector = ShotDetector()
        self.shot_classifier = ShotClassifier()
        self.technique = TechniqueAnalyzer()
        self.swing = SwingAnalyzer()
        self.footwork = FootworkAnalyzer()
        self.phase_detector = PhaseDetector()
        
        # Experimental phase-entry correction.
        # Legacy mode remains unchanged.
        self.phase_detector.experimental_stance_downswing_entry = (
            self.angle_geometry_mode == "pixel"
        )

        # Require a confirmed impact event before the
# batting sequence enters Follow Through.
        self.phase_detector.follow_through_requires_impact = True

        # Bat-Ball Contact Analyzer
        self.bat_ball_contact = BatBallContactAnalyzer(
            contact_distance=3.0,
            near_contact_distance=35.0
        )

        # Impact Detector
        self.impact = ImpactDetector(
            impact_distance=3.0,
            confirmation_frames=2,
            cooldown_frames=15,
            rearm_distance=15,
            min_bat_confidence=0.10,
            min_ball_confidence=0.15,
            follow_through_grace_frames=8,
            debug=True
        )

        # Impact State
        self.impact_locked = False
        self.last_impact_report = None
        self.impact_shot = "Unknown"
        self.impact_phase = "Unknown"
        self.impact_score = 0.0
        self.impact_frame = None
        self.impact_display_frames = 15
        self.impact_display_counter = 0

        # Temporal Analysis State
        self.impact_context = deque(maxlen=16)
        self.frame_number = 0
        self.last_phase = "Waiting"
        self.last_shot = "Unknown"
        self.last_angles = {}
        self.last_scores = {}
        self.last_overall_score = 0.0
        self.last_landmarks = None

        self.phase_sequence_active = False
        self.fresh_ball_streak = 0

        # Two fresh detections are enough to establish that
        # a real delivery is present while rejecting a
        # one-frame false detection.
        self.phase_activation_frames = 2

        self.phase_activation_frame = None
        self.phase_observations = {}

        # Diagnostic State
        self.last_analysis_status = "Starting"
        self.status_counts = {}

        print("=" * 60)
        print("        CRICKET ANALYZER READY")
        print("=" * 60)
        print()

    def empty_report(self, ball=None):
        return {
            "frame_number": self.frame_number,
            "impact_frame": self.impact_frame,
            "phase": "Waiting",
            "shot": "Unknown",
            "angles": {},
            "scores": {},
            "overall_score": 0,
            "technique_score": 0.0,
            "mistakes": [],
            "ball": ball,
            "swing": {
                "speed": "Unknown",
                "direction": "Unknown",
                "timing": "Unknown"
            },
            "footwork": [],
            "impact": {
                "detected": False,
                "event": False,
                "display": False,
                "confidence": 0,
                "distance": None,
                "wrist_speed": 0.0,
                "ball_speed": 0.0,
                "status": "Not Detected"
            },
            "bat_ball_contact": {
                "valid": False,
                "contact": False,
                "near_contact": False,
                "distance": None,
                "iou": 0.0,
                "bat_confidence": 0.0,
                "ball_confidence": 0.0,
                "bat_center": None,
                "ball_center": None,
                "status": "No Bat/Ball"
            },
            "analysis_status": "Waiting",
            "batsman_detected": False,
            "pose_detected": False,
            "landmarks_detected": False,
            "ball_detected": ball is not None,
            "ball_predicted": False,
            "tracking": False
        }

    def set_status(self, status):
        self.last_analysis_status = str(status)
        self.status_counts[self.last_analysis_status] = (
            self.status_counts.get(self.last_analysis_status, 0) + 1
        )

    def safe_float(self, value, default=0.0):
        try:
            number = float(value)
            if math.isfinite(number):
                return number
        except Exception:
            pass
        return default

    def safe_point(self, point):
        if point is None:
            return None
        try:
            if len(point) < 2:
                return None
            x = self.safe_float(point[0])
            y = self.safe_float(point[1])
            return (int(round(x)), int(round(y)))
        except Exception:
            return None
    
    # =====================================================
    # JOINT MEASUREMENT COVERAGE REPORT
    # =====================================================

    def build_joint_coverage_report(
        self,
        angles,
        landmarks,
        pose_results
    ):
        from analysis.joint_config import JOINTS
        from analysis.angle_calculator import get_point

        angles = angles if isinstance(angles, dict) else {}
        landmarks = (
            landmarks if isinstance(landmarks, dict) else {}
        )

        measured = [
            name for name in JOINTS
            if name in angles
        ]

        missing = [
            name for name in JOINTS
            if name not in angles
        ]

        raw_pose = getattr(
            pose_results,
            "pose_landmarks",
            None
        )

        missing_details = {}

        for joint in missing:

            failed_landmarks = []

            for name in JOINTS[joint]:

                if get_point(landmarks, name) is not None:
                    continue

                visibility = None
                reason = "Landmark not retained"

                try:
                    index = (
                        self.pose.mp_pose.PoseLandmark[
                            name.upper()
                        ].value
                    )

                    raw_landmark = (
                        raw_pose.landmark[index]
                    )

                    visibility = float(
                        raw_landmark.visibility
                    )

                    if (
                        math.isfinite(visibility)
                        and visibility < 0.40
                    ):
                        reason = "Filtered: low visibility"

                    elif name in landmarks:
                        reason = "Invalid coordinates"

                except (
                    AttributeError,
                    KeyError,
                    IndexError,
                    TypeError,
                    ValueError
                ):
                    pass

                failed_landmarks.append({
                    "name": name,
                    "reason": reason,
                    "visibility": (
                        round(visibility, 3)
                        if visibility is not None
                        and math.isfinite(visibility)
                        else None
                    )
                })

            missing_details[joint] = {
                "reason": (
                    "Required landmark unavailable"
                    if failed_landmarks
                    else "Angle calculation unavailable"
                ),
                "failed_landmarks": failed_landmarks
            }

        total = len(JOINTS)

        return {
            "status": (
                "Complete"
                if not missing
                else "Incomplete measurements"
            ),
            "expected_joints": total,
            "measured_joints": len(measured),
            "coverage_percent": round(
                len(measured) / total * 100,
                1
            ) if total else 0.0,
            "missing_joints": missing,
            "missing_details": missing_details,
            "note": (
                "Coverage reports available joint angles, "
                "not pose accuracy or scoring confidence. "
                "Existing scores are unchanged."
            )
        }
    

    def clamp_box(self, box, frame_shape):
        if box is None:
            return None
        try:
            frame_height, frame_width = frame_shape[:2]
            if frame_width <= 0 or frame_height <= 0:
                return None
            x1, y1, x2, y2 = box
            x1 = max(0, min(int(round(self.safe_float(x1))), frame_width - 1))
            y1 = max(0, min(int(round(self.safe_float(y1))), frame_height - 1))
            x2 = max(0, min(int(round(self.safe_float(x2))), frame_width))
            y2 = max(0, min(int(round(self.safe_float(y2))), frame_height))
            if x2 <= x1 or y2 <= y1:
                return None
            return (x1, y1, x2, y2)
        except Exception:
            return None

    def get_best_detection(self, detections, class_name, min_confidence=0.0):
        if not detections:
            return None
        target = str(class_name).lower().strip()
        candidates = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            detected_name = str(detection.get("class_name", "")).lower().strip()
            if detected_name != target:
                continue
            confidence = self.safe_float(detection.get("confidence", 0.0))
            if confidence < min_confidence:
                continue
            candidates.append(detection)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: self.safe_float(item.get("confidence", 0.0))
        )

    def box_iou(self, box_a, box_b):
        if box_a is None or box_b is None:
            return 0.0
        try:
            ax1, ay1, ax2, ay2 = box_a
            bx1, by1, bx2, by2 = box_b
            intersection_x1 = max(ax1, bx1)
            intersection_y1 = max(ay1, by1)
            intersection_x2 = min(ax2, bx2)
            intersection_y2 = min(ay2, by2)
            intersection_width = max(0, intersection_x2 - intersection_x1)
            intersection_height = max(0, intersection_y2 - intersection_y1)
            intersection_area = intersection_width * intersection_height
            if intersection_area <= 0:
                return 0.0
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            union_area = area_a + area_b - intersection_area
            if union_area <= 0:
                return 0.0
            return intersection_area / union_area
        except Exception:
            return 0.0

    def get_bat_inside_batsman(self, detections, batsman_box, min_confidence=0.10):
        if not detections or batsman_box is None:
            return None
        try:
            bx1, by1, bx2, by2 = batsman_box
            batsman_width = bx2 - bx1
            batsman_height = by2 - by1
            if batsman_width <= 0 or batsman_height <= 0:
                return None
            batsman_area = batsman_width * batsman_height

            expand_x = int(batsman_width * 0.75)
            expand_y = int(batsman_height * 0.35)
            expanded_x1 = max(0, bx1 - expand_x)
            expanded_y1 = max(0, by1 - expand_y)
            expanded_x2 = bx2 + expand_x
            expanded_y2 = by2 + expand_y

            valid_candidates = []
            for detection in detections:
                if not isinstance(detection, dict):
                    continue
                class_name = str(detection.get("class_name", "")).lower().strip()
                confidence = self.safe_float(detection.get("confidence", 0.0))
                if class_name != "bat" or confidence < min_confidence:
                    continue
                bat_box = self.get_detection_box(detection)
                bat_box = self.clamp_box(bat_box, (expanded_y2 + 1, expanded_x2 + 1))
                if bat_box is None:
                    continue
                x1, y1, x2, y2 = bat_box
                bat_width = x2 - x1
                bat_height = y2 - y1
                if bat_width <= 0 or bat_height <= 0:
                    continue
                bat_area = bat_width * bat_height
                if bat_area > (batsman_area * 0.60):
                    continue

                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                if not (expanded_x1 <= center_x <= expanded_x2 and expanded_y1 <= center_y <= expanded_y2):
                    continue

                aspect_ratio = bat_width / max(bat_height, 1)
                if aspect_ratio > 4.0:
                    continue

                batsman_center_x = (bx1 + bx2) / 2.0
                batsman_center_y = (by1 + by2) / 2.0
                distance = math.sqrt((center_x - batsman_center_x) ** 2 + (center_y - batsman_center_y) ** 2)
                normalized_distance = distance / max(math.sqrt(batsman_area), 1.0)
                distance_penalty = normalized_distance * 0.25
                selection_score = confidence * 3.0 - distance_penalty

                candidate = detection.copy()
                candidate["box"] = bat_box
                candidate["bat_center"] = (int(round(center_x)), int(round(center_y)))
                candidate["_selection_score"] = selection_score
                valid_candidates.append(candidate)

            if not valid_candidates:
                return None
            return max(valid_candidates, key=lambda item: item.get("_selection_score", 0.0))
        except Exception as e:
            print(f"[Frame {self.frame_number}] Bat filtering error: {e}")
            return None

    def get_detection_box(self, detection):
        if detection is None:
            return None
        if isinstance(detection, dict):
            return detection.get("box")
        return detection

    def draw_single_detection(self, frame, detection, color, label):
        if frame is None or detection is None:
            return frame
        try:
            box = self.get_detection_box(detection)
            box = self.clamp_box(box, frame.shape)
            if box is None:
                return frame
            x1, y1, x2, y2 = box
            confidence = self.safe_float(detection.get("confidence", 0.0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{label} {confidence:.2f}",
                (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
            )
        except Exception:
            pass
        return frame

    def get_pixel_landmark(self, landmarks, name, crop_box, crop_shape):
        if landmarks is None or name not in landmarks:
            return None
        try:
            lm = landmarks[name]
            nx = self.safe_float(lm.get("x", 0.0))
            ny = self.safe_float(lm.get("y", 0.0))
            cx1, cy1, cx2, cy2 = crop_box
            crop_h, crop_w = crop_shape[:2]
            px = cx1 + nx * crop_w
            py = cy1 + ny * crop_h
            return (int(round(px)), int(round(py)))
        except Exception:
            return None

    def midpoint(self, p1, p2):
        if p1 is None or p2 is None:
            return None
        try:
            return (int(round((p1[0] + p2[0]) / 2.0)), int(round((p1[1] + p2[1]) / 2.0)))
        except Exception:
            return None

    def point_distance(self, p1, p2):
        if p1 is None or p2 is None:
            return float('inf')
        try:
            return math.sqrt((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2)
        except Exception:
            return float('inf')

    def create_csrt_tracker(self):
        try:
            if hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
            elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
                return cv2.legacy.TrackerCSRT_create()
        except Exception:
            pass
        return None

    def estimate_bat_contact_point(self, landmarks, crop_box, crop_shape, ball):
        if landmarks is None or ball is None:
            return None, None

        ball_center = self.safe_point(ball.get("center") if isinstance(ball, dict) else ball)
        if ball_center is None:
            return None, None

        left_wrist = self.get_pixel_landmark(landmarks, "left_wrist", crop_box, crop_shape)
        right_wrist = self.get_pixel_landmark(landmarks, "right_wrist", crop_box, crop_shape)
        left_elbow = self.get_pixel_landmark(landmarks, "left_elbow", crop_box, crop_shape)
        right_elbow = self.get_pixel_landmark(landmarks, "right_elbow", crop_box, crop_shape)
        left_shoulder = self.get_pixel_landmark(landmarks, "left_shoulder", crop_box, crop_shape)
        right_shoulder = self.get_pixel_landmark(landmarks, "right_shoulder", crop_box, crop_shape)

        candidates = []
        wrist_mid = self.midpoint(left_wrist, right_wrist)
        if wrist_mid is not None:
            distance = self.point_distance(wrist_mid, ball_center)
            candidates.append((distance, wrist_mid, "both"))

        for wrist, elbow, shoulder, side in [
            (left_wrist, left_elbow, left_shoulder, "left"),
            (right_wrist, right_elbow, right_shoulder, "right")
        ]:
            if wrist is not None and elbow is not None:
                dx = wrist[0] - elbow[0]
                dy = wrist[1] - elbow[1]
                length = math.sqrt(dx * dx + dy * dy)
                if length > 5:
                    arm_scale = self.point_distance(shoulder, wrist) if shoulder else length
                    extension = max(40.0, arm_scale * 0.6)  # Dynamic reach scaling
                    contact = (
                        wrist[0] + (dx / length) * extension,
                        wrist[1] + (dy / length) * extension
                    )
                    distance = self.point_distance(contact, ball_center)
                    candidates.append((distance, contact, side))

        if not candidates:
            return None, None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1], candidates[0][2]

    def estimate_bat_angle(self, landmarks, crop_box, crop_shape, arm_side):
        if arm_side == "left":
            wrist_name = "left_wrist"
            elbow_name = "left_elbow"
        elif arm_side == "right":
            wrist_name = "right_wrist"
            elbow_name = "right_elbow"
        else:
            left_wrist = self.get_pixel_landmark(landmarks, "left_wrist", crop_box, crop_shape)
            right_wrist = self.get_pixel_landmark(landmarks, "right_wrist", crop_box, crop_shape)
            left_elbow = self.get_pixel_landmark(landmarks, "left_elbow", crop_box, crop_shape)
            right_elbow = self.get_pixel_landmark(landmarks, "right_elbow", crop_box, crop_shape)
            wrist_mid = self.midpoint(left_wrist, right_wrist)
            elbow_mid = self.midpoint(left_elbow, right_elbow)
            if wrist_mid is None or elbow_mid is None:
                return None
            dx = wrist_mid[0] - elbow_mid[0]
            dy = wrist_mid[1] - elbow_mid[1]
            if abs(dx) < 1 and abs(dy) < 1:
                return None
            return math.degrees(math.atan2(abs(dy), abs(dx)))

        wrist = self.get_pixel_landmark(landmarks, wrist_name, crop_box, crop_shape)
        elbow = self.get_pixel_landmark(landmarks, elbow_name, crop_box, crop_shape)
        if wrist is None or elbow is None:
            return None
        dx = wrist[0] - elbow[0]
        dy = wrist[1] - elbow[1]
        if abs(dx) < 1 and abs(dy) < 1:
            return None
        return math.degrees(math.atan2(abs(dy), abs(dx)))

    def refine_shot(self, shot, phase, landmarks, crop_box, crop_shape, arm_side):
        if shot is None:
            return "Unknown"
        shot_string = str(shot).lower()
        bat_angle = self.estimate_bat_angle(landmarks, crop_box, crop_shape, arm_side)
        if bat_angle is None:
            return shot
        vertical_bat = (bat_angle >= 65)
        valid_phase = phase in ["Downswing", "Impact", "Follow Through"]
        if "pull" in shot_string and valid_phase and vertical_bat:
            return "Defensive Shot"
        if "controlled" in shot_string and vertical_bat and phase in ["Impact", "Follow Through"]:
            return "Defensive Shot"
        return shot

    def valid_impact_phase(self, phase):
        if phase is None:
            return False
        return str(phase).strip().lower() in {"downswing", "impact"}

    def detect_impact(self, bat_detection, ball, phase):
        empty_impact = {
            "detected": False,
            "event": False,
            "display": False,
            "confidence": 0,
            "distance": None,
            "wrist_speed": 0.0,
            "ball_speed": 0.0,
            "status": "Not Detected",
            "impact_count": getattr(self.impact, "impact_count", 0),
            "cooldown": getattr(self.impact, "cooldown_remaining", 0)
        }
        self.impact_context.append({
            "frame": self.frame_number, "shot": self.last_shot,
            "phase": str(phase), "score": self.last_overall_score
        })
        try:
            result = self.impact.analyze(
                bat_detection, ball, phase, frame_number=self.frame_number
            )
        except Exception as e:
            self.set_status("Impact Error")
            print(f"[Frame {self.frame_number}] Impact detector error: {e}")
            return empty_impact

        if not isinstance(result, dict):
            return empty_impact

        new_event = bool(result.get("event", False))
        impact_report = result.copy()
        impact_report["event"] = new_event
        impact_report["detected"] = new_event

        if new_event:
            impact_report["display"] = True
            self.last_impact_report = impact_report.copy()
            self.impact_locked = True
            self.impact_frame = result["impact_frame"]
            context = next((item for item in self.impact_context
                            if item["frame"] == self.impact_frame), None)
            self.impact_shot = context["shot"] if context else "Unknown"
            self.impact_phase = context["phase"] if context else "Unknown"
            self.impact_score = context["score"] if context else 0.0
            self.impact_display_counter = self.impact_display_frames
            return impact_report

        if self.impact_display_counter > 0:
            self.impact_display_counter -= 1
            if self.last_impact_report is not None:
                display_report = result.copy()
                display_report["impact_frame"] = self.impact_frame
                display_report["confirmation_frame"] = self.last_impact_report.get("confirmation_frame")
                display_report["detected"] = False
                display_report["event"] = False
                display_report["display"] = True
                display_report["status"] = "Impact Display"
                display_report["cooldown"] = result.get("cooldown", 0)
                return display_report

        impact_report["display"] = False
        impact_report["detected"] = False
        impact_report["event"] = False
        return impact_report

    def reset_impact_if_needed(self, ball, batsman_box):
        if not self.impact_locked or ball is None or batsman_box is None:
            return
        center = self.safe_point(ball.get("center") if isinstance(ball, dict) else ball)
        if center is None:
            return
        bx, by = center
        x1, y1, x2, y2 = batsman_box
        closest_x = max(x1, min(bx, x2))
        closest_y = max(y1, min(by, y2))
        distance = math.sqrt((bx - closest_x) ** 2 + (by - closest_y) ** 2)
        if distance > 250:
            self.impact_locked = False
            self.last_impact_report = None
            self.impact_display_counter = 0

    def draw_contact_point(self, frame, contact_point, impact_report):
        if frame is None:
            return frame
        point = self.safe_point(contact_point)
        if point is None:
            return frame
        h, w = frame.shape[:2]
        x, y = point
        if not (0 <= x < w and 0 <= y < h):
            return frame

        detected = False
        display = False
        if isinstance(impact_report, dict):
            detected = bool(impact_report.get("detected", False))
            display = bool(impact_report.get("display", False))

        if detected:
            cv2.circle(frame, (x, y), 14, (0, 0, 255), -1)
            cv2.putText(frame, f"Impact est. frame {impact_report.get('impact_frame')}", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        elif display:
            cv2.circle(frame, (x, y), 10, (0, 165, 255), 2)
        else:
            cv2.circle(frame, (x, y), 7, (0, 255, 255), 2)
        return frame

    def draw_analysis_overlay(self, frame, report):
        if frame is None or not isinstance(report, dict):
            return frame
        h, w = frame.shape[:2]
        phase = report.get("phase", "Waiting")
        shot = report.get("shot", "Unknown")
        score = self.safe_float(report.get("overall_score", 0))
        status = report.get("analysis_status", "Unknown")
        ball = report.get("ball")
        impact = report.get("impact", {})
        impact_detected = bool(impact.get("detected", False)) if isinstance(impact, dict) else False

        panel_w = min(440, w - 20)
        panel_h = 205
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)

        y = 38
        cv2.putText(frame, f"Phase : {phase}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 32
        cv2.putText(frame, f"Shot  : {shot}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 32
        cv2.putText(frame, f"Score : {score:.1f}/100", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 32
        ball_status = "Detected" if ball is not None else "Not Detected"
        cv2.putText(frame, f"Ball  : {ball_status}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        y += 32
        impact_text = f"est. frame {impact.get('impact_frame')}" if impact.get("display") else "NO"
        cv2.putText(frame, f"Impact: {impact_text}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, ((0, 0, 255) if impact_detected else (255, 255, 255)), 2, cv2.LINE_AA)
        y += 32
        cv2.putText(frame, f"Status: {status}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Frame: {self.frame_number}", (max(10, w - 180), h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def process_frame(self, frame, source_frame_number=None):
        # Source indices are one-based. Existing callers remain valid if every
        # decoded frame is passed exactly once, from the start of the video.
        if source_frame_number is None:
            self.frame_number += 1
        else:
            if (isinstance(source_frame_number, bool)
                    or not isinstance(source_frame_number, int)
                    or source_frame_number <= self.frame_number):
                raise ValueError("source_frame_number must increase; reset after seeking")
            self.frame_number = source_frame_number
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            report = self.empty_report()
            report["impact"] = self.detect_impact(None, None, "Unknown")
            return frame, report

        output = frame.copy()
        report = self.empty_report()
        self.set_status("Processing")

        # Step 1: Cricket Object Detection
        try:
            cricket_detections = self.cricket_detector.detect(frame)
            if cricket_detections is None:
                cricket_detections = []
        except Exception as e:
            cricket_detections = []
            print(f"[Frame {self.frame_number}] Cricket detector error: {e}")

        # Step 2: Batsman Detection
        batsman_detection = self.get_best_detection(cricket_detections, "batsman", min_confidence=0.15)
        box = self.get_detection_box(batsman_detection)
        box = self.clamp_box(box, frame.shape)

        if box is None:
            try:
                fallback = self.detector.detect_batsman(frame)
                box = self.get_detection_box(fallback) if isinstance(fallback, dict) else fallback
                box = self.clamp_box(box, frame.shape)
            except Exception as e:
                box = None

        if box is None:
            self.set_status("Batsman Not Detected")
            report["analysis_status"] = "Batsman Not Detected"
            report["impact"] = self.detect_impact(None, None, "Unknown")
            output = self.draw_analysis_overlay(output, report)
            return output, report

        report["batsman_detected"] = True
        if batsman_detection is not None:
            output = self.draw_single_detection(output, batsman_detection, (255, 255, 0), "BATSMAN")
        else:
            try:
                output = self.detector.draw_batsman(output, box)
            except Exception:
                x1, y1, x2, y2 = box
                cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Step 3: Bat Detection + CSRT Tracking
        bat_detection = None
        try:
            bat_detection = self.get_bat_inside_batsman(cricket_detections, batsman_box=box, min_confidence=0.10)
        except Exception as e:
            bat_detection = None

        if bat_detection is not None:
            bat_box = self.get_detection_box(bat_detection)
            bat_box = self.clamp_box(bat_box, frame.shape)
            if bat_box is not None:
                self.last_valid_bat_box = bat_box
                self.last_valid_bat_confidence = self.safe_float(bat_detection.get("confidence", 0.0))
                self.frames_since_last_bat = 0

                self.bat_tracker = self.create_csrt_tracker()
                if self.bat_tracker is not None:
                    x1, y1, x2, y2 = bat_box
                    tracker_box = (int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1)))
                    try:
                        self.bat_tracker.init(frame, tracker_box)
                        self.bat_tracker_active = True
                    except Exception:
                        self.bat_tracker_active = False
        else:
            self.frames_since_last_bat += 1
            if self.frames_since_last_bat <= self.max_bat_missing_frames and self.bat_tracker_active and self.bat_tracker is not None:
                try:
                    success, tracked_box = self.bat_tracker.update(frame)
                    if success:
                        tx, ty, tw, th = [int(v) for v in tracked_box]
                        tracked_bat_box = self.clamp_box((tx, ty, tx + tw, ty + th), frame.shape)
                        if tracked_bat_box is not None:
                            self.last_valid_bat_box = tracked_bat_box
                            bat_detection = {
                                "box": tracked_bat_box,
                                "confidence": self.last_valid_bat_confidence,
                                "class_name": "bat",
                                "tracker_only": True
                            }
                except Exception:
                    self.bat_tracker_active = False
            else:
                self.bat_tracker_active = False

        # Step 4: Ball Detection & Tracking
        raw_ball = None
        try:
            raw_ball = self.ball_detector.detect_ball(frame)
        except Exception:
            raw_ball = None

        impact_ball = deepcopy(raw_ball) if isinstance(raw_ball, dict) else None

        tracked_ball = None
        try:
            tracked_ball = self.ball_tracker.update(raw_ball)
        except Exception:
            tracked_ball = None

        is_predicted_ball = False
        if isinstance(tracked_ball, dict):
            is_predicted_ball = bool(tracked_ball.get("predicted", False) or tracked_ball.get("is_predicted", False))

        real_ball = tracked_ball if (tracked_ball is not None and not is_predicted_ball) else None
        # Raw current-frame boxes preserve the observed contact timing. Smoothed
        # or predicted positions remain available for visualization only.
        # =========================================================
        # DELIVERY CONTEXT / PHASE ACTIVATION
        # =========================================================
        #
        # IMPORTANT:
        #
        # We use RAW current-frame ball detection here.
        #
        # Predicted/tracked-only ball positions must NOT start
        # the batting phase sequence.
        #
        # One isolated detection is also not enough.
        # We require multiple consecutive fresh observations.
        # =========================================================

        fresh_ball_for_phase = False

        if isinstance(
            impact_ball,
            dict
        ):

            try:
                ball_confidence = float(
                    impact_ball.get(
                        "confidence",
                        0.0
                    )
                )

            except Exception:
                ball_confidence = 0.0


            try:
                required_ball_confidence = float(
                    getattr(
                        self.impact,
                        "min_ball_confidence",
                        0.15
                    )
                )

            except Exception:
                required_ball_confidence = 0.15


            fresh_ball_for_phase = (
                ball_confidence
                >=
                required_ball_confidence
            )


        # ---------------------------------------------------------
        # FRESH BALL STREAK
        # ---------------------------------------------------------

        if fresh_ball_for_phase:

            self.fresh_ball_streak += 1

        else:

            self.fresh_ball_streak = 0


        # ---------------------------------------------------------
        # ACTIVATE REAL BATTING SEQUENCE
        # ---------------------------------------------------------

        if (
            not self.phase_sequence_active
            and
            self.fresh_ball_streak
            >=
            self.phase_activation_frames
        ):

            self.phase_sequence_active = True

            self.phase_activation_frame = (
                self.frame_number
            )


            print(
                f"[PHASE GATE] "
                f"Delivery context confirmed at "
                f"frame {self.frame_number}"
            )


        
            # =====================================================
            # RESET PHASE DETECTOR FOR REAL DELIVERY
            # =====================================================

            try:

                if hasattr(
                    self.phase_detector,
                    "reset"
                ):

                    self.phase_detector.reset()

                    print(
                        "[PHASE GATE] "
                        "Phase detector reset"
                    )

            except Exception as e:

                print(
                    f"[PHASE RESET ERROR] {e}"
                )


            # =====================================================
            # ARM DELIVERY ENTRY
            # =====================================================

            try:

                if hasattr(
                    self.phase_detector,
                    "arm_delivery_entry"
                ):

                    self.phase_detector.arm_delivery_entry()

                else:

                    print(
                        "[PHASE ENTRY ERROR] "
                        "arm_delivery_entry() not found"
                    )

            except Exception as e:

                print(
                    f"[PHASE ENTRY ERROR] {e}"
                )

            self.last_phase = "Waiting"
            self.last_shot = "Preparing"

        # Step 5: Bat-Ball Contact Diagnostic
        bat_ball_report = {"valid": False, "contact": False, "near_contact": False, "distance": None, "iou": 0.0, "status": "No Bat/Ball"}
        try:
            bat_ball_report = self.bat_ball_contact.analyze(bat_detection, impact_ball)
        except Exception:
            pass

        report["bat_ball_contact"] = bat_ball_report
        report["ball"] = tracked_ball
        report["ball_detected"] = real_ball is not None
        report["ball_predicted"] = is_predicted_ball
        report["tracking"] = real_ball is not None and bool(real_ball.get("tracked", False))

        if tracked_ball is not None:
            try:
                output = self.ball_detector.draw_ball(output, tracked_ball)
            except Exception:
                pass

        try:
            output = self.ball_tracker.draw(output)
        except Exception:
            pass

        self.reset_impact_if_needed(real_ball, box)

        try:
            self.shot_detector.update(tracked_ball, box)
        except Exception:
            pass

        # Step 6: Crop Batsman
        crop = None
        crop_box = None
        try:
            crop, crop_box = self.detector.crop_batsman(frame, box)
        except Exception:
            pass

        if crop is None or crop_box is None:
            self.set_status("Crop Failed")
            report["analysis_status"] = "Crop Failed"
            report["impact"] = self.detect_impact(
                bat_detection,
                impact_ball,
                self.last_phase
            )
            output = self.draw_analysis_overlay(output, report)
            return output, report

        # Step 7 & 8: Pose & Landmarks
        pose_results = None
        try:
            pose_results = self.pose.detect_pose(crop)
        except Exception:
            pose_results = None

        if pose_results is None:
            self.set_status("Pose Failed")
            report["analysis_status"] = "Pose Failed"
            report["impact"] = self.detect_impact(
                bat_detection,
                impact_ball,
                self.last_phase
            )
            output = self.draw_analysis_overlay(output, report)
            return output, report

        try:
            self.pose.draw_pose(crop, pose_results)
        except Exception:
            pass

        landmarks = None
        try:
            landmarks = self.pose.get_landmarks(pose_results)
        except Exception:
            landmarks = None

        if not landmarks:
            self.set_status("No Landmarks")
            report["analysis_status"] = "No Landmarks"
            report["impact"] = self.detect_impact(
                bat_detection,
                impact_ball,
                self.last_phase
            )
            output = self.draw_analysis_overlay(output, report)
            return output, report

        report["landmarks_detected"] = True
        self.last_landmarks = landmarks

        # Step 9 & 10: Joint Angles & Smoothing
        try:
            
            angles = calculate_joint_angles(
                landmarks,
                image_shape=(
                    crop.shape
                    if self.angle_geometry_mode == "pixel"
                    else None
                )
            )

        except Exception:
            angles = {}

        smoothed_angles = {}
        for joint, angle in angles.items():
            try:
                smoothed_angles[joint] = self.smoother.smooth(joint, angle)
            except Exception:
                smoothed_angles[joint] = angle
        angles = smoothed_angles
        self.last_angles = angles.copy()
        joint_coverage_report = None

        if self.joint_coverage_enabled:
            joint_coverage_report = (
                self.build_joint_coverage_report(
                    angles,
                    landmarks,
                    pose_results
                )
            )
        
        # =====================================================
        # EXPERIMENTAL JOINT AVAILABILITY DIAGNOSTIC
        # =====================================================
        # Read-only: does not alter landmarks, angles,
        # scoring, phase detection, or impact detection.

        if (
            os.getenv("CRICKET_JOINT_DEBUG", "0") == "1"
            and self.phase_sequence_active
            and self.frame_number in (35, 123)
        ):

            from analysis.joint_config import JOINTS

            expected_joints = set(JOINTS.keys())
            measured_joints = set(angles.keys())

            missing_joints = sorted(
                expected_joints - measured_joints
            )

            print(
                f"\n[JOINT DIAG] Frame={self.frame_number}"
            )

            print(
                "[JOINT DIAG] "
                f"Measured={len(expected_joints & measured_joints)}"
                f"/{len(expected_joints)}"
            )

            print(
                f"[JOINT DIAG] Missing={missing_joints}"
            )

            raw_pose = getattr(
                pose_results,
                "pose_landmarks",
                None
            )

            if raw_pose is not None:

                required_landmarks = sorted({
                    name
                    for joint in missing_joints
                    for name in JOINTS[joint]
                })

                for name in required_landmarks:

                    index = self.pose.mp_pose.PoseLandmark[
                        name.upper()
                    ].value

                    raw_landmark = raw_pose.landmark[index]

                    visibility = float(
                        raw_landmark.visibility
                    )

                    retained = name in landmarks

                    print(
                        f"[JOINT DIAG] {name}: "
                        f"visibility={visibility:.3f} "
                        f"retained={retained} "
                        f"below_threshold={visibility < 0.40}"
                    )

        
        # =====================================================
        # DIAGNOSTIC: NORMALIZED VS PIXEL-SCALED ANGLES
        # =====================================================
        # No changes to production angles or scores.
        # Compare only the confirmed impact frames
        # of our two regression videos.

        if (
            self.phase_sequence_active
            and self.frame_number in (35, 123)
        ):

            try:
                from analysis.angle_calculator import (
                    calculate_angle,
                    get_point
                )
                from analysis.joint_config import JOINTS

                crop_height, crop_width = crop.shape[:2]

                print(
                    f"\n[ANGLE GEOMETRY DIAGNOSTIC] "
                    f"Frame={self.frame_number} "
                    f"Crop={crop_width}x{crop_height}"
                )

                for joint_name, (a, b, c) in JOINTS.items():

                    p1 = get_point(landmarks, a)
                    p2 = get_point(landmarks, b)
                    p3 = get_point(landmarks, c)

                    if any(
                        p is None
                        for p in (p1, p2, p3)
                    ):
                        continue

                    # Existing raw normalized-coordinate angle
                    normalized_angle = calculate_angle(
                        p1, p2, p3
                    )

                    # Diagnostic pixel-scaled angle
                    pixel_points = [
                        (p[0] * crop_width, p[1] * crop_height)
                        for p in (p1, p2, p3)
                    ]

                    corrected_angle = calculate_angle(
                        *pixel_points
                    )

                    if (
                        normalized_angle is None
                        or corrected_angle is None
                    ):
                        continue

                    difference = round(
                        corrected_angle - normalized_angle,
                        2
                    )

                    print(
                        f"[ANGLE GEOMETRY] {joint_name}: "
                        f"normalized={normalized_angle:.2f} "
                        f"pixel_scaled={corrected_angle:.2f} "
                        f"difference={difference:+.2f}"
                    )

            except Exception as e:
                print(
                    f"[ANGLE GEOMETRY WARNING] {e}"
                )

        
        # =====================================================
        # DIAGNOSTIC: NORMALIZED VS PIXEL-SCALED ANGLES
        # =====================================================
        # No changes to production angles or scores.
        # Compare only the confirmed impact frames
        # of our two regression videos.

        if (
            self.phase_sequence_active
            and self.frame_number in (35, 123)
        ):

            try:
                from analysis.angle_calculator import (
                    calculate_angle,
                    get_point
                )
                from analysis.joint_config import JOINTS

                crop_height, crop_width = crop.shape[:2]

                

                for joint_name, (a, b, c) in JOINTS.items():

                    p1 = get_point(landmarks, a)
                    p2 = get_point(landmarks, b)
                    p3 = get_point(landmarks, c)

                    if any(
                        p is None
                        for p in (p1, p2, p3)
                    ):
                        continue

                    # Existing raw normalized-coordinate angle
                    normalized_angle = calculate_angle(
                        p1, p2, p3
                    )

                    # Diagnostic pixel-scaled angle
                    pixel_points = [
                        (p[0] * crop_width, p[1] * crop_height)
                        for p in (p1, p2, p3)
                    ]

                    corrected_angle = calculate_angle(
                        *pixel_points
                    )

                    if (
                        normalized_angle is None
                        or corrected_angle is None
                    ):
                        continue

                    difference = round(
                        corrected_angle - normalized_angle,
                        2
                    )

                    

            except Exception as e:
                print(
                    f"[ANGLE GEOMETRY WARNING] {e}"
                )


        # Step 11: Scoring
        try:
            
            batting_stance = getattr(
                self.footwork,
                "batting_stance",
                "Right-handed"
            )

            scores = calculate_score(
                angles,
                batting_stance=batting_stance
            )

            mistakes = self.technique.analyze(angles)
        except Exception:
            scores = {}
            mistakes = []

        numeric_scores = [float(v) for v in scores.values() if math.isfinite(float(v))] if isinstance(scores, dict) else []
        overall = round(sum(numeric_scores) / len(numeric_scores), 1) if numeric_scores else 0.0
        technique_score = calculate_technique_score(scores, mistakes)
        self.last_scores = scores.copy()
        self.last_overall_score = overall
        # =========================================================
        # Step 12: Phase
        # =========================================================

        # ---------------------------------------------------------
        # BEFORE DELIVERY ACTIVATION
        # ---------------------------------------------------------
        #
        # Do NOT allow ordinary body/arm movement to generate:
        #
        # Backlift -> Downswing -> Follow Through -> Recovery
        #
        # before an actual delivery has been established.
        # ---------------------------------------------------------

        if not self.phase_sequence_active:

            phase = "Waiting"

            self.last_phase = phase


        # ---------------------------------------------------------
        # REAL DELIVERY ACTIVE
        # ---------------------------------------------------------

        else:

            try:

                phase = self.phase_detector.update(
                    angles
                )

            except Exception:

                phase = self.last_phase


            if (
                not phase
                or
                str(phase).strip() == ""
            ):

                phase = self.last_phase


            self.last_phase = str(
                phase
            )

       

        # =========================================================
        # Step 13: Shot Classification
        # =========================================================

        if not self.phase_sequence_active:

            shot = "Preparing"

        else:

            try:

                shot = self.shot_classifier.classify(
                    angles,
                    phase
                )

            except Exception:

                shot = self.last_shot


            if (
                not shot
                or
                str(shot).strip() == ""
            ):

                shot = self.last_shot

        # Step 14 - 16: Technique, Footwork, Swing
        
        try:

            batting_stance = getattr(
                self.footwork,
                "batting_stance",
                "Unknown"
            )

            mistakes = self.technique.analyze(
                angles,
                batting_stance=batting_stance,
                feedback_mode=self.technique_feedback_mode
            )

        except Exception as e:

            print(f"[TECHNIQUE WARNING] {e}")
            mistakes = []
            
        # =====================================================
        # STANCE-AWARE TECHNIQUE SCORE SYNCHRONIZATION
        # =====================================================
        # Use the same mistakes that appear in the final report.

        try:
            technique_score = calculate_technique_score(
                scores,
                mistakes
            )

        except Exception as e:
            print(f"[TECHNIQUE SCORE WARNING] {e}")
            technique_score = 0.0


        try:
            footwork = self.footwork.analyze(
                landmarks,
                angles,
                frame_number=self.frame_number,
                crop_box=crop_box
            )
        except Exception as e:
            print(f"[FOOTWORK ERROR] {e}")

            footwork = {
                "feedback": ["Footwork analysis unavailable"],
                "footwork_score": 0
            }    


        # Swing analysis runs after Step 18,
# once impact synchronization is complete.

        # Step 17: Contact Point & Refinement
        contact_point, arm_side = None, None
        try:
            contact_point, arm_side = self.estimate_bat_contact_point(landmarks, crop_box, crop.shape, real_ball)
        except Exception:
            pass

        shot = self.refine_shot(shot, phase, landmarks, crop_box, crop.shape, arm_side)
        self.last_shot = str(shot)

        # Step 18: Impact Detection
        impact_report = self.detect_impact(bat_detection, impact_ball, phase)
        new_impact_event = (
            isinstance(impact_report, dict)
            and
            bool(
                impact_report.get(
                    "event",
                    False
                )
            )
        )

        impact_is_current_frame = (
            new_impact_event
            and
            impact_report.get(
                "impact_frame"
            ) == self.frame_number
        )


        if impact_is_current_frame:

            # -----------------------------------------------------
            # Synchronize phase
            # -----------------------------------------------------

            try:
                phase = (
                    self.phase_detector.register_impact()
                )

            except Exception as e:

                print(
                    f"[Frame {self.frame_number}] "
                    f"Phase impact sync error: {e}"
                )

                # Safe report-level fallback.
                phase = "Impact"


            self.last_phase = str(
                phase
            )


            # -----------------------------------------------------
            # A confirmed bat-ball contact means the shot has
            # physically been played.
            # -----------------------------------------------------

            shot = "Shot Played"

            self.last_shot = str(
                shot
            )


            # -----------------------------------------------------
            # Correct impact-frame snapshot
            #
            # detect_impact() initially stores the provisional
            # pre-impact phase/shot. Replace them with the
            # synchronized values.
            # -----------------------------------------------------

            self.impact_phase = str(
                phase
            )

            self.impact_shot = str(
                shot
            )

            self.impact_score = float(
                overall
            )


            # -----------------------------------------------------
            # Correct the temporal context for this exact frame.
            # -----------------------------------------------------

            try:

                if (
                    self.impact_context
                    and
                    self.impact_context[-1].get(
                        "frame"
                    ) == self.frame_number
                ):

                    self.impact_context[-1][
                        "phase"
                    ] = str(
                        phase
                    )

                    self.impact_context[-1][
                        "shot"
                    ] = str(
                        shot
                    )

                    self.impact_context[-1][
                        "score"
                    ] = float(
                        overall
                    )

            except Exception:
                pass


            # -----------------------------------------------------
            # Recalculate swing report with synchronized phase.
            # -----------------------------------------------------
        
        # =========================================================
        # Step 19: Swing Analysis
        # AFTER impact synchronization
        # =========================================================

        # Exactly one swing update per processed source frame.
        # The phase now includes any confirmed-impact
        # synchronization performed in Step 18.
        
        # =====================================================
        # PHASE COVERAGE: RECORD FINAL FRAME LABEL
        # =====================================================

        if self.phase_sequence_active:

            self.phase_observations[self.frame_number] = {
                "phase": str(phase).strip().lower(),
                "left_wrist_present": (
                    landmarks.get("left_wrist") is not None
                ),
                "right_wrist_present": (
                    landmarks.get("right_wrist") is not None
                )
            }


        try:
            swing_report = self.swing.analyze(
                landmarks,
                phase,
                frame_number=self.frame_number
            )

        except Exception as e:
            print(
                f"[SWING ERROR] Frame "
                f"{self.frame_number}: {e}"
            )

            swing_report = {
                "speed": "Unknown",
                "direction": "Unknown",
                "timing": "Unknown"
            }

        # Step 20: Put Pose Crop Back & Redraw Overlays
    


        # Step 20: Put Pose Crop Back & Redraw Overlays
        try:
            x1, y1, x2, y2 = crop_box
            fh, fw = output.shape[:2]
            x1, x2 = max(0, min(int(x1), fw)), max(0, min(int(x2), fw))
            y1, y2 = max(0, min(int(y1), fh)), max(0, min(int(y2), fh))
            if x2 > x1 and y2 > y1:
                resized_crop = cv2.resize(crop, (x2 - x1, y2 - y1))
                output[y1:y2, x1:x2] = resized_crop
        except Exception:
            pass

        if batsman_detection is not None:
            output = self.draw_single_detection(output, batsman_detection, (255, 255, 0), "BATSMAN")
        else:
            try:
                output = self.detector.draw_batsman(output, box)
            except Exception:
                pass

        if bat_detection is not None:
            output = self.draw_single_detection(output, bat_detection, (0, 255, 0), "BAT")

        if tracked_ball is not None:
            try:
                output = self.ball_detector.draw_ball(output, tracked_ball)
            except Exception:
                pass

        try:
            output = self.ball_tracker.draw(output)
        except Exception:
            pass

        output = self.draw_contact_point(output, contact_point, impact_report)

        self.set_status("Analysis Working")
        report = {
            "frame_number": self.frame_number,
            "phase": phase,
            "shot": shot,
            "angles": angles,
            "scores": scores,
            "overall_score": overall,
            "technique_score": technique_score,
            "impact_frame": self.impact_frame,
            "impact_shot": self.impact_shot,
            "impact_phase": self.impact_phase,
            "impact_score": self.impact_score,
            "mistakes": mistakes,
            "ball": tracked_ball,
            "bat_ball_contact": bat_ball_report,
            "swing": swing_report,
            "footwork": footwork,
            "impact": impact_report,
            "analysis_status": "Analysis Working",
            "batsman_detected": True,
            "pose_detected": True,
            "landmarks_detected": True,
            "ball_detected": tracked_ball is not None,
            "tracking": tracked_ball is not None and bool(tracked_ball.get("tracked", False))
        }
        if joint_coverage_report is not None:
            report["joint_measurement_reliability"] = (
                joint_coverage_report
            )

        output = self.draw_analysis_overlay(output, report)
        return output, report
        
    
    
    # =========================================================
    # PHASE COVERAGE AND RELIABILITY DIAGNOSTICS
    # =========================================================

    def get_phase_coverage(self):

        phase_names = (
            "backlift",
            "downswing",
            "impact",
            "follow through",
            "recovery"
        )

        def normalize_phase(value):
            label = str(value or "").strip().lower()
            label = label.replace("-", " ")
            label = label.replace("_", " ")
            return " ".join(label.split())

        # ================================================
        # 1. COLLECT OBSERVED PHASE FRAMES
        # ================================================

        observations = getattr(
            self, "phase_observations", {}
        )

        observed = {
            name: {}
            for name in phase_names
        }

        other_phases = {}

        for frame_number, data in observations.items():

            phase = normalize_phase(data.get("phase"))

            if phase in observed:
                observed[phase][frame_number] = data

            else:
                other_phases[phase] = (
                    other_phases.get(phase, 0) + 1
                )

        # ================================================
        # 2. COLLECT ACCEPTED WRIST-MOTION SAMPLES
        # ================================================

        motion_frames = getattr(
            self.swing,
            "motion_sample_frames",
            []
        )

        motion_phases = getattr(
            self.swing,
            "motion_sample_phases",
            []
        )

        accepted = {
            name: set()
            for name in phase_names
        }

        for frame_number, phase in zip(
            motion_frames,
            motion_phases
        ):

            phase = normalize_phase(phase)

            if phase in accepted:
                accepted[phase].add(frame_number)

        # ================================================
        # 3. EVALUATE EACH PHASE
        # ================================================

        phase_results = {}

        for phase in phase_names:

            phase_frames = observed[phase]

            observed_frames = set(
                phase_frames.keys()
            )

            accepted_frames = accepted[phase]

            matched_frames = (
                observed_frames & accepted_frames
            )

            observed_count = len(observed_frames)

            sample_count = len(matched_frames)

            wrist_frames = sum(
                1
                for data in phase_frames.values()
                if (
                    data.get("left_wrist_present")
                    or data.get("right_wrist_present")
                )
            )

            missing_samples = (
                observed_count - sample_count
            )

            # ============================================
            # 4. COVERAGE STATUS
            # ============================================

            if observed_count == 0:

                status = "Not observed"

                explanation = (
                    "No analyzed frame received this "
                    "phase label."
                )

            elif sample_count == 0:

                status = "No qualifying motion samples"

                if wrist_frames == 0:

                    explanation = (
                        "The phase was observed, but "
                        "neither wrist landmark was "
                        "available in those frames."
                    )

                else:

                    explanation = (
                        "The phase was observed and "
                        "wrist landmarks were present, "
                        "but no temporal motion sample "
                        "qualified. Frame gaps, phase "
                        "transitions, or tracking "
                        "continuity may be involved."
                    )

            elif sample_count < 3:

                status = "Limited evidence"

                explanation = (
                    "Only one or two qualifying "
                    "wrist-motion samples were recorded."
                )

            else:

                status = "Measured"

                explanation = (
                    "At least three qualifying "
                    "wrist-motion samples were recorded."
                )

            # ============================================
            # 5. PHASE REPORT
            # ============================================

            phase_results[phase] = {
                "status": status,
                "observed_frames": observed_count,
                "wrist_landmark_frames": wrist_frames,
                "qualifying_motion_samples": sample_count,
                "frames_without_motion_sample": (
                    missing_samples
                ),
                "motion_sample_coverage_percent": (
                    round(
                        sample_count / observed_count * 100,
                        1
                    )
                    if observed_count > 0
                    else None
                ),
                "first_observed_frame": (
                    min(observed_frames)
                    if observed_frames
                    else None
                ),
                "last_observed_frame": (
                    max(observed_frames)
                    if observed_frames
                    else None
                ),
                "unmatched_motion_sample_frames": sorted(
                    accepted_frames - observed_frames
                ),
                "explanation": explanation
            }

        # ================================================
        # 6. RETURN COMPLETE COVERAGE REPORT
        # ================================================

        return {
            "status": (
                "Evaluated"
                if observations
                else "No qualifying phase observations"
            ),
            "delivery_activation_frame": (
                self.phase_activation_frame
            ),
            "phase_observation_frames": len(
                observations
            ),
            "other_observed_phases": other_phases,
            "phases": phase_results,
            "note": (
                "Coverage is measured only for frames "
                "that reached post-impact-sync phase "
                "analysis after delivery activation. "
                "Frames failing earlier detection or "
                "pose stages are excluded. Missing "
                "samples do not prove missing physical "
                "movement. The report does not alter "
                "phase logic, measurements, or scores."
            )
        }


    def reset(self):
        self.impact_context = deque(maxlen=16)
        self.frame_number = 0
        self.last_phase = "Waiting"
        self.last_shot = "Unknown"
        self.phase_sequence_active = False
        self.fresh_ball_streak = 0
        self.phase_activation_frame = None
        self.phase_observations = {}
        self.last_angles = {}
        self.last_scores = {}
        self.last_overall_score = 0.0
        self.last_landmarks = None
        self.last_analysis_status = "Reset"
        self.status_counts = {}

        try:
            self.ball_tracker.clear()
        except Exception:
            pass

        self.bat_tracker = None
        self.bat_tracker_active = False
        self.last_valid_bat_box = None
        self.last_valid_bat_confidence = 0.0
        self.frames_since_last_bat = 0

        try:
            if hasattr(self.smoother, "clear"):
                self.smoother.clear()
        except Exception:
            pass

        try:
            if hasattr(self.phase_detector, "reset"):
                self.phase_detector.reset()
        except Exception as e:
            print(
                    f"[PHASE ENTRY ERROR] {e}"
                )
            pass

        

        self.impact_locked = False
        self.last_impact_report = None
        self.impact_shot = "Unknown"
        self.impact_phase = "Unknown"
        self.impact_score = 0.0
        self.impact_frame = None
        self.impact_display_counter = 0

        try:
            if hasattr(self.impact, "reset"):
                self.impact.reset()
        except Exception:
            pass

        try:
            if hasattr(self.ball_detector, "clear"):
                self.ball_detector.clear()
        except Exception:
            pass

        print("Cricket Analyzer reset successfully.")
