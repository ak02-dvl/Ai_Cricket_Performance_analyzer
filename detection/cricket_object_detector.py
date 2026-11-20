import cv2
from ultralytics import YOLO
class CricketObjectDetector:
    def __init__(
        self,
        model_path="models/cricket_model.pt",
        confidence=0.05,
        bat_confidence=0.20,
        batsman_confidence=0.15,
        ball_confidence=0.05
    ):
        # =====================================================
        # LOAD MODEL
        # =====================================================
        self.model = YOLO(model_path)
        self.default_confidence = confidence
        # Separate confidence thresholds
        self.bat_confidence = bat_confidence
        self.batsman_confidence = batsman_confidence
        self.ball_confidence = ball_confidence
        # =====================================================
        # CLASS NAMES
        # =====================================================
        self.class_names = self.model.names
        if not isinstance(self.class_names, dict):
            self.class_names = {
                i: name
                for i, name in enumerate(self.class_names)
            }
        # =====================================================
        # CLASS IDS
        # =====================================================
        self.class_ids = {}
        for class_id, name in self.class_names.items():
            name_lower = str(name).lower().strip()
            self.class_ids[name_lower] = int(class_id)
        print("\nCricket Object Detector")
        print("-" * 50)
        for class_id, name in self.class_names.items():
            print(f"{class_id} -> {name}")
        print("-" * 50)
        print(f"General confidence : {self.default_confidence}")
        print(f"Batsman confidence : {self.batsman_confidence}")
        print(f"Bat confidence     : {self.bat_confidence}")
        print(f"Ball confidence    : {self.ball_confidence}")
        print("-" * 50)
        # =====================================================
        # TEMPORAL MEMORY
        # =====================================================
        self.last_batsman = None
        self.last_bat = None
        self.last_ball = None
        self.batsman_missed = 0
        self.bat_missed = 0
        self.ball_missed = 0
        # =====================================================
        # PERSISTENCE
        # =====================================================
        self.batsman_max_missed = 12
        self.bat_max_missed = 8
        self.ball_max_missed = 5
        # =====================================================
        # BAT TRACKING SETTINGS
        # =====================================================
        # Maximum allowed movement of the bat between frames.
        # This prevents sudden false detections on the bowler.
        self.max_bat_movement = 400
        # Maximum allowed distance between bat and batsman.
        # This is calculated using the batsman bounding-box size.
        self.bat_region_width_ratio = 0.70
        self.bat_region_height_ratio = 0.35
    # =========================================================
    # BOX AREA
    # =========================================================
    def box_area(self, box):
        x1, y1, x2, y2 = box
        width = max(
            0,
            x2 - x1
        )
        height = max(
            0,
            y2 - y1
        )
        return width * height
    # =========================================================
    # BOX CENTER
    # =========================================================
    def box_center(self, box):
        x1, y1, x2, y2 = box
        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0
        )
    # =========================================================
    # CENTER DISTANCE
    # =========================================================
    def center_distance(
        self,
        box1,
        box2
    ):
        c1 = self.box_center(box1)
        c2 = self.box_center(box2)
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        return (
            dx * dx +
            dy * dy
        ) ** 0.5
    # =========================================================
    # CENTER INSIDE BOX
    # =========================================================
    def center_inside_box(
        self,
        small_box,
        large_box
    ):
        cx, cy = self.box_center(
            small_box
        )
        x1, y1, x2, y2 = large_box
        return (
            x1 <= cx <= x2
            and
            y1 <= cy <= y2
        )
    # =========================================================
    # IOU
    # =========================================================
    def calculate_iou(
        self,
        box1,
        box2
    ):
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
        intersection_width = max(
            0,
            x2 - x1
        )
        intersection_height = max(
            0,
            y2 - y1
        )
        intersection = (
            intersection_width *
            intersection_height
        )
        area1 = self.box_area(box1)
        area2 = self.box_area(box2)
        union = (
            area1 +
            area2 -
            intersection
        )
        if union <= 0:
            return 0.0
        return (
            intersection / union
        )
    # =========================================================
    # BAT NEAR BATSMAN
    # =========================================================
    def is_bat_near_batsman(
        self,
        bat_box,
        batsman_box
    ):
        bx1, by1, bx2, by2 = batsman_box
        batsman_width = bx2 - bx1
        batsman_height = by2 - by1
        # Expand the batsman bounding box.
        #
        # The bat can be outside the person's bounding box,
        # especially when the batsman extends the bat.
        expanded_box = (
            bx1 - int(
                batsman_width *
                self.bat_region_width_ratio
            ),
            by1 - int(
                batsman_height *
                self.bat_region_height_ratio
            ),
            bx2 + int(
                batsman_width *
                self.bat_region_width_ratio
            ),
            by2 + int(
                batsman_height *
                self.bat_region_height_ratio
            )
        )
        bat_center_x, bat_center_y = self.box_center(
            bat_box
        )
        ex1, ey1, ex2, ey2 = expanded_box
        return (
            ex1 <= bat_center_x <= ex2
            and
            ey1 <= bat_center_y <= ey2
        )
    # =========================================================
    # BAT MOVEMENT VALIDATION
    # =========================================================
    def is_bat_movement_valid(
        self,
        current_bat_box
    ):
        # No previous bat available.
        if self.last_bat is None:
            return True
        previous_bat_box = self.last_bat["box"]
        movement = self.center_distance(
            current_bat_box,
            previous_bat_box
        )
        return movement <= self.max_bat_movement
    # =========================================================
    # DETECT
    # =========================================================
    def detect(self, frame):
    # =====================================================
    # YOLO INFERENCE
    # =====================================================
        results = self.model.predict(
            frame,
            conf=min(self.default_confidence, self.bat_confidence, self.batsman_confidence, self.ball_confidence),
            iou=0.50,
            max_det=50,
            verbose=False
        )
        if not results:
            return self.use_previous_detections()
        result = results[0]
        if result.boxes is None:
            return self.use_previous_detections()
        # =====================================================
        # RAW DETECTIONS
        # =====================================================
        raw_detections = []
        for box in result.boxes:
            try:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                coordinates = (
                    box.xyxy[0]
                    .tolist()
                )
                x1, y1, x2, y2 = coordinates
                x1 = int(round(x1))
                y1 = int(round(y1))
                x2 = int(round(x2))
                y2 = int(round(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                class_name = str(
                    self.class_names.get(
                        cls_id,
                        cls_id
                    )
                )
                class_name_lower = (
                    class_name
                    .lower()
                    .strip()
                )
                raw_detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "class_name_lower": class_name_lower,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2)
                })
            except Exception:
                continue
        # =====================================================
        # CLASS FILTERING
        # =====================================================
        batsmen = []
        bats = []
        balls = []
        for detection in raw_detections:
            name = detection["class_name_lower"]
            confidence = detection["confidence"]
            if name == "batsman":
                if confidence >= self.batsman_confidence:
                    batsmen.append(detection)
            elif name == "bat":
                if confidence >= self.bat_confidence:
                    bats.append(detection)
            elif name == "ball":
                if confidence >= self.ball_confidence:
                    balls.append(detection)
        # =====================================================
        # SELECT BATSMAN
        # =====================================================
        selected_batsman = None
        if batsmen:
            if self.last_batsman is not None:
                selected_batsman = min(
                    batsmen,
                    key=lambda d:
                    self.center_distance(
                        d["box"],
                        self.last_batsman["box"]
                    )
                )
            else:
                selected_batsman = max(
                    batsmen,
                    key=lambda d:
                    d["confidence"]
                )
        # =====================================================
        # SELECT BAT
        # =====================================================
        selected_bat = None
        valid_bats = []
        if selected_batsman is not None:
            batsman_box = selected_batsman["box"]
            bx1, by1, bx2, by2 = batsman_box
            batsman_width = bx2 - bx1
            batsman_height = by2 - by1
            batsman_area = self.box_area(
                batsman_box
            )
            # Same expansion used by diagnostic
            expanded_x1 = int(
                bx1 - batsman_width * 0.75
            )
            expanded_y1 = int(
                by1 - batsman_height * 0.35
            )
            expanded_x2 = int(
                bx2 + batsman_width * 0.75
            )
            expanded_y2 = int(
                by2 + batsman_height * 0.35
            )
            for bat in bats:
                bat_box = bat["box"]
                candidate_area = self.box_area(
                    bat_box
                )
                # ---------------------------------------------
                # FILTER 1: BAT TOO LARGE
                # ---------------------------------------------
                if candidate_area > batsman_area * 0.60:
                    continue
                # ---------------------------------------------
                # FILTER 2: BAT CENTER NEAR BATSMAN
                # ---------------------------------------------
                center_x, center_y = self.box_center(
                    bat_box
                )
                center_inside = (
                    expanded_x1 <= center_x <= expanded_x2
                    and
                    expanded_y1 <= center_y <= expanded_y2
                )
                if not center_inside:
                    continue
                # ---------------------------------------------
                # FILTER 3: REQUIRE OVERLAP
                # ---------------------------------------------
                cx1, cy1, cx2, cy2 = bat_box
                overlap_x1 = max(
                    cx1,
                    expanded_x1
                )
                overlap_y1 = max(
                    cy1,
                    expanded_y1
                )
                overlap_x2 = min(
                    cx2,
                    expanded_x2
                )
                overlap_y2 = min(
                    cy2,
                    expanded_y2
                )
                overlap_width = max(
                    0,
                    overlap_x2 - overlap_x1
                )
                overlap_height = max(
                    0,
                    overlap_y2 - overlap_y1
                )
                overlap_area = (
                    overlap_width *
                    overlap_height
                )
                if overlap_area <= 0:
                    continue
                # ---------------------------------------------
                # FILTER 4: REJECT EXTREMELY WIDE BOXES
                # ---------------------------------------------
                bat_width = max(
                    1,
                    cx2 - cx1
                )
                bat_height = max(
                    1,
                    cy2 - cy1
                )
                aspect_ratio = (
                    bat_width /
                    bat_height
                )
                if aspect_ratio > 4.0:
                    continue
                # ---------------------------------------------
                # FILTER 5: MOVEMENT VALIDATION
                # ---------------------------------------------
                if not self.is_bat_movement_valid(
                    bat_box
                ):
                    continue
                valid_bats.append(bat)
        # =====================================================
        # SCORE VALID BAT CANDIDATES
        # =====================================================
        if valid_bats:
            if selected_batsman is not None:
                batsman_box = selected_batsman["box"]
                scored_bats = []
                for bat in valid_bats:
                    distance = self.center_distance(
                        bat["box"],
                        batsman_box
                    )
                    confidence_score = (
                        bat["confidence"] * 100.0
                    )
                    distance_score = max(
                        0.0,
                        100.0 - distance / 10.0
                    )
                    total_score = (
                        confidence_score * 0.65
                        +
                        distance_score * 0.35
                    )
                    scored_bats.append(
                        (
                            bat,
                            total_score
                        )
                    )
                selected_bat = max(
                    scored_bats,
                    key=lambda item: item[1]
                )[0]
            elif self.last_bat is not None:
                selected_bat = min(
                    valid_bats,
                    key=lambda d:
                    self.center_distance(
                        d["box"],
                        self.last_bat["box"]
                    )
                )
            else:
                selected_bat = max(
                    valid_bats,
                    key=lambda d:
                    d["confidence"]
                )
        # =====================================================
        # SELECT BALL
        # =====================================================
        selected_ball = None
        if balls:
            if self.last_ball is not None:
                selected_ball = min(
                    balls,
                    key=lambda d:
                    self.center_distance(
                        d["box"],
                        self.last_ball["box"]
                    )
                )
            else:
                selected_ball = max(
                    balls,
                    key=lambda d:
                    d["confidence"]
                )
        # =====================================================
        # MEMORY
        # =====================================================
        if selected_batsman is not None:
            self.last_batsman = selected_batsman
            self.batsman_missed = 0
        else:
            self.batsman_missed += 1
        if selected_bat is not None:
            self.last_bat = selected_bat
            self.bat_missed = 0
        else:
            self.bat_missed += 1
        if selected_ball is not None:
            self.last_ball = selected_ball
            self.ball_missed = 0
        else:
            self.ball_missed += 1
        # =====================================================
        # CLEAR EXPIRED MEMORY
        # =====================================================
        if self.batsman_missed > self.batsman_max_missed:
            self.last_batsman = None
        if self.bat_missed > self.bat_max_missed:
            self.last_bat = None
        if self.ball_missed > self.ball_max_missed:
            self.last_ball = None
        # =====================================================
        # FINAL DETECTIONS
        # =====================================================
        final_detections = []
        if self.last_batsman is not None:
            detection = self.last_batsman.copy()
            detection["tracked"] = (
                selected_batsman is None
            )
            final_detections.append(
                detection
            )
        if self.last_bat is not None:
            detection = self.last_bat.copy()
            detection["tracked"] = (
                selected_bat is None
            )
            final_detections.append(
                detection
            )
        if self.last_ball is not None:
            detection = self.last_ball.copy()
            detection["tracked"] = (
                selected_ball is None
            )
            final_detections.append(
                detection
            )
        return final_detections
    # =========================================================
    # PREVIOUS DETECTIONS
    # =========================================================
    def use_previous_detections(self):
        final_detections = []
        # =====================================================
        # BATSMAN
        # =====================================================
        self.batsman_missed += 1
        if (
            self.last_batsman is not None
            and
            self.batsman_missed
            <=
            self.batsman_max_missed
        ):
            detection = (
                self.last_batsman.copy()
            )
            detection["tracked"] = True
            final_detections.append(
                detection
            )
        # =====================================================
        # BAT
        # =====================================================
        self.bat_missed += 1
        if (
            self.last_bat is not None
            and
            self.bat_missed
            <=
            self.bat_max_missed
        ):
            detection = (
                self.last_bat.copy()
            )
            detection["tracked"] = True
            final_detections.append(
                detection
            )
        # =====================================================
        # BALL
        # =====================================================
        self.ball_missed += 1
        if (
            self.last_ball is not None
            and
            self.ball_missed
            <=
            self.ball_max_missed
        ):
            detection = (
                self.last_ball.copy()
            )
            detection["tracked"] = True
            final_detections.append(
                detection
            )
        # =====================================================
        # CLEAR EXPIRED MEMORY
        # =====================================================
        if (
            self.batsman_missed
            >
            self.batsman_max_missed
        ):
            self.last_batsman = None
        if (
            self.bat_missed
            >
            self.bat_max_missed
        ):
            self.last_bat = None
        if (
            self.ball_missed
            >
            self.ball_max_missed
        ):
            self.last_ball = None
        return final_detections
    # =========================================================
    # DRAW DETECTIONS
    # =========================================================
    def draw_detections(
        self,
        frame,
        detections
    ):
        output = frame.copy()
        for detection in detections:
            class_name = detection.get(
                "class_name",
                "Unknown"
            )
            confidence = float(
                detection.get(
                    "confidence",
                    0.0
                )
            )
            box = detection.get(
                "box"
            )
            if box is None:
                continue
            x1, y1, x2, y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            name = (
                class_name
                .lower()
                .strip()
            )
            # =================================================
            # COLORS
            # =================================================
            if name == "batsman":
                color = (
                    255,
                    255,
                    0
                )
            elif name == "bat":
                color = (
                    255,
                    255,
                    255
                )
            elif name == "ball":
                color = (
                    0,
                    255,
                    0
                )
            else:
                color = (
                    255,
                    255,
                    255
                )
            # =================================================
            # TRACKING LABEL
            # =================================================
            tracked = detection.get(
                "tracked",
                False
            )
            if tracked:
                label = (
                    f"{class_name} "
                    f"{confidence:.2f} "
                    f"(tracking)"
                )
            else:
                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )
            # =================================================
            # DRAW BOX
            # =================================================
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )
            # =================================================
            # LABEL
            # =================================================
            text_y = max(
                20,
                y1 - 8
            )
            cv2.putText(
                output,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA
            )
        return output
