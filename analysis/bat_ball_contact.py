"""
Bat-Ball Contact Analyzer
-------------------------

Uses YOLO bat and ball bounding boxes to estimate
the geometric proximity between the bat and ball.

This module does NOT decide the final impact event.
It provides a reliable bat-ball contact signal that
CricketAnalyzer can combine with:

    - batting phase
    - ball confidence
    - bat confidence
    - temporal movement

The important distinction is:

    distance <= contact threshold
        = contact candidate

    contact candidate + valid phase + temporal confirmation
        = impact event
"""

import math


class BatBallContactAnalyzer:

    def __init__(
        self,
        contact_distance=35.0,
        near_contact_distance=70.0
    ):

        self.contact_distance = float(
            contact_distance
        )

        self.near_contact_distance = float(
            near_contact_distance
        )

    # =========================================================
    # SAFE BOX
    # =========================================================

    def safe_box(self, detection):

        if detection is None:
            return None

        if not isinstance(
            detection,
            dict
        ):
            return None

        box = detection.get("box")

        if box is None:
            return None

        try:

            x1 = float(box[0])
            y1 = float(box[1])
            x2 = float(box[2])
            y2 = float(box[3])

        except Exception:

            return None

        if x2 < x1:
            x1, x2 = x2, x1

        if y2 < y1:
            y1, y2 = y2, y1

        if x2 <= x1 or y2 <= y1:
            return None

        return (
            x1,
            y1,
            x2,
            y2
        )

    # =========================================================
    # BOX CENTER
    # =========================================================

    def box_center(self, box):

        if box is None:
            return None

        x1, y1, x2, y2 = box

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0
        )

    # =========================================================
    # POINT TO RECTANGLE DISTANCE
    # =========================================================

    def point_to_box_distance(
        self,
        point,
        box
    ):

        if point is None or box is None:
            return None

        px, py = point

        x1, y1, x2, y2 = box

        closest_x = max(
            x1,
            min(px, x2)
        )

        closest_y = max(
            y1,
            min(py, y2)
        )

        dx = px - closest_x
        dy = py - closest_y

        return math.sqrt(
            dx * dx +
            dy * dy
        )

    # =========================================================
    # BOX TO BOX DISTANCE
    # =========================================================

    def box_distance(
        self,
        box1,
        box2
    ):

        if box1 is None or box2 is None:
            return None

        x1, y1, x2, y2 = box1
        a1, b1, a2, b2 = box2

        horizontal_gap = max(
            a1 - x2,
            x1 - a2,
            0.0
        )

        vertical_gap = max(
            b1 - y2,
            y1 - b2,
            0.0
        )

        return math.sqrt(
            horizontal_gap ** 2 +
            vertical_gap ** 2
        )

    # =========================================================
    # OVERLAP
    # =========================================================

    def calculate_iou(
        self,
        box1,
        box2
    ):

        if box1 is None or box2 is None:
            return 0.0

        x1 = max(
            box1[0],
            box2[0]
        )

        y1 = max(
            box1[1],
            box2[1]
        )

        x2 = min(
            box1[2],
            box2[2]
        )

        y2 = min(
            box1[3],
            box2[3]
        )

        width = max(
            0.0,
            x2 - x1
        )

        height = max(
            0.0,
            y2 - y1
        )

        intersection = (
            width * height
        )

        area1 = (
            max(
                0.0,
                box1[2] - box1[0]
            )
            *
            max(
                0.0,
                box1[3] - box1[1]
            )
        )

        area2 = (
            max(
                0.0,
                box2[2] - box2[0]
            )
            *
            max(
                0.0,
                box2[3] - box2[1]
            )
        )

        union = (
            area1 +
            area2 -
            intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    # =========================================================
    # ANALYZE
    # =========================================================

    def analyze(
        self,
        bat_detection,
        ball_detection
    ):

        result = {

            "valid": False,

            "contact": False,

            "near_contact": False,

            "distance": None,

            "iou": 0.0,

            "bat_confidence": 0.0,

            "ball_confidence": 0.0,

            "bat_center": None,

            "ball_center": None,

            "status":
                "Bat or ball not detected"
        }

        bat_box = self.safe_box(
            bat_detection
        )

        ball_box = self.safe_box(
            ball_detection
        )

        if bat_box is None:
            return result

        if ball_box is None:
            return result

        # -----------------------------------------------------
        # Confidence
        # -----------------------------------------------------

        try:

            bat_confidence = float(
                bat_detection.get(
                    "confidence",
                    0.0
                )
            )

        except Exception:

            bat_confidence = 0.0

        try:

            ball_confidence = float(
                ball_detection.get(
                    "confidence",
                    0.0
                )
            )

        except Exception:

            ball_confidence = 0.0

        # -----------------------------------------------------
        # Centers
        # -----------------------------------------------------

        bat_center = self.box_center(
            bat_box
        )

        ball_center = self.box_center(
            ball_box
        )

        # -----------------------------------------------------
        # Geometric distance
        #
        # This is the shortest distance between the
        # two bounding rectangles.
        # -----------------------------------------------------

        distance = self.box_distance(
            bat_box,
            ball_box
        )

        iou = self.calculate_iou(
            bat_box,
            ball_box
        )

        if distance is None:
            return result

        # -----------------------------------------------------
        # Contact classification
        # -----------------------------------------------------

        contact = (
            distance <=
            self.contact_distance
        )

        near_contact = (
            distance <=
            self.near_contact_distance
        )

        if contact:

            status = "BAT-BALL CONTACT"

        elif near_contact:

            status = "NEAR BAT-BALL CONTACT"

        else:

            status = "BAT-BALL SEPARATED"

        result = {

            "valid": True,

            "contact": contact,

            "near_contact":
                near_contact,

            "distance":
                round(
                    distance,
                    2
                ),

            "iou":
                round(
                    iou,
                    4
                ),

            "bat_confidence":
                round(
                    bat_confidence,
                    3
                ),

            "ball_confidence":
                round(
                    ball_confidence,
                    3
                ),

            "bat_center":
                bat_center,

            "ball_center":
                ball_center,

            "status":
                status
        }

        return result