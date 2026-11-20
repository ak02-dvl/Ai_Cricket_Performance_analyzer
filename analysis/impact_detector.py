"""Conservative image-space impact estimate; proximity alone is never an event.

Uses two observed steps before and after a candidate. Confirmation is delayed,
but impact_frame identifies the candidate, not the confirmation frame.
Thresholds require validation against annotated video; this is not proof of contact.
"""
import math
from collections import deque


class ImpactDetector:
    REVISION = "isolated-outlier-recovery-v4"
    def __init__(self, impact_distance=3.0, confirmation_frames=2,
                 cooldown_frames=15, rearm_distance=15.0,
                 min_bat_confidence=0.15, min_ball_confidence=0.25,
                 min_ball_speed=2.0, direction_change_degrees=35.0,
                 speed_change_ratio=0.5, follow_through_grace_frames=8, debug=False):
        self.impact_distance = float(impact_distance)
        self.confirmation_frames = max(2, int(confirmation_frames))
        self.cooldown_frames = max(0, int(cooldown_frames))
        self.rearm_distance = max(float(rearm_distance), self.impact_distance + 1)
        self.min_bat_confidence = float(min_bat_confidence)
        self.min_ball_confidence = float(min_ball_confidence)
        self.min_ball_speed = float(min_ball_speed)
        self.direction_change_degrees = float(direction_change_degrees)
        self.speed_change_ratio = float(speed_change_ratio)
        self.follow_through_grace_frames = max(0, int(follow_through_grace_frames))
        self.debug = bool(debug)
        self.reset()

    def reset(self):
        self.history = deque(maxlen=2 * self.confirmation_frames + 1)
        self.last_frame = None
        self.last_swing_phase_frame = None
        self.last_impact_frame = None
        self.cooldown_remaining = 0
        self.rearmed = True
        self.impact_count = 0
        self.impact_detected = False

    @staticmethod
    def _box(detection, min_confidence):
        if not isinstance(detection, dict):
            return None
        # Tracker output is for display, not independent contact evidence.
        if any(detection.get(k, False) for k in
               ('predicted', 'is_predicted', 'tracked', 'tracker_only')):
            return None
        try:
            confidence = float(detection.get('confidence', 0))
            box = tuple(float(v) for v in detection['box'])
            if not math.isfinite(confidence) or not min_confidence <= confidence <= 1:
                return None
            if len(box) != 4 or not all(math.isfinite(v) for v in box):
                return None
            if box[2] <= box[0] or box[3] <= box[1]:
                return None
            return box
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def box_distance(a, b):
        return math.hypot(max(a[0]-b[2], b[0]-a[2], 0),
                          max(a[1]-b[3], b[1]-a[3], 0))
    @staticmethod
    def box_intersection_area(a, b):
        if a is None or b is None:
            return 0.0

        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)

        return width * height

    @staticmethod
    def _center(box):
        return ((box[0]+box[2])/2, (box[1]+box[3])/2)

    @staticmethod
    def _velocity(a, b):
        dt = b['frame'] - a['frame']
        return tuple((y-x)/dt for x, y in zip(a['ball'], b['ball']))

    @staticmethod
    def _cos(a, b):
        scale = math.hypot(*a) * math.hypot(*b)
        return max(-1.0, min(1.0, sum(x*y for x, y in zip(a, b))/scale)) if scale else -1.0

    def analyze(self, bat_detection, ball, phase='Unknown', frame_number=None):

    # =========================================================
    # FRAME VALIDATION
    # =========================================================

        if frame_number is None:
            frame_number = 1 if self.last_frame is None else self.last_frame + 1

        if (
            isinstance(frame_number, bool)
            or not isinstance(frame_number, int)
            or frame_number < 1
        ):
            raise ValueError(
                'frame_number must be a positive, one-based integer'
            )

        if self.last_frame is not None and frame_number <= self.last_frame:
            raise ValueError(
                'Frame numbers must increase; '
                'call reset() after seeking/restarting'
            )

        # If too many source frames were skipped, old motion history
        # is no longer trustworthy.
        if (
            self.last_frame is not None
            and frame_number > self.last_frame + 2
        ):
            self.history.clear()

        self.last_frame = frame_number


        # =========================================================
        # PHASE STATE
        # =========================================================

        normalized_phase = str(phase).strip().lower()

        if normalized_phase in ('downswing', 'impact'):
            self.last_swing_phase_frame = frame_number

        elif normalized_phase != 'follow through':
            self.last_swing_phase_frame = None

        phase_eligible = (
            normalized_phase in ('downswing', 'impact')
            or (
                normalized_phase == 'follow through'
                and self.last_swing_phase_frame is not None
                and (
                    frame_number - self.last_swing_phase_frame
                    <= self.follow_through_grace_frames
                )
            )
        )


        # =========================================================
        # BASIC IMPACT STATE
        # =========================================================

        self.impact_detected = False

        if self.last_impact_frame is not None:
            self.cooldown_remaining = max(
                0,
                self.cooldown_frames
                - (frame_number - self.last_impact_frame)
            )
        else:
            self.cooldown_remaining = 0

        result = dict(
            detected=False,
            event=False,
            confidence=0.0,
            confidence_kind='uncalibrated evidence score',
            distance=None,
            wrist_speed=0.0,
            ball_speed=0.0,
            impact_count=self.impact_count,
            cooldown=self.cooldown_remaining,
            impact_frame=None,
            confirmation_frame=None,
            overlap=False,
            overlap_area=0.0,
            status='Insufficient fresh detections'
        )


        # =========================================================
        # VALIDATE CURRENT BAT / BALL DETECTIONS
        # =========================================================

        bat_box = self._box(
            bat_detection,
            self.min_bat_confidence
        )

        ball_box = self._box(
            ball,
            self.min_ball_confidence
        )

        result['input_phase'] = str(phase)

        result['bat_input'] = self._input_summary(
            bat_detection,
            bat_box,
            self.min_bat_confidence
        )

        result['ball_input'] = self._input_summary(
            ball,
            ball_box,
            self.min_ball_confidence
        )


        # Remove stale history.
        if (
            self.history
            and frame_number - self.history[-1]['frame'] > 2
        ):
            self.history.clear()


        # A real ball detection is required.
        if ball_box is None:

            result['status'] = (
                'Rejected input: bat='
                + result['bat_input']['reason']
                + ', ball='
                + result['ball_input']['reason']
            )

            return self._finish(result)


        # =========================================================
        # BAT-BALL DISTANCE
        # =========================================================

        distance = (
            self.box_distance(bat_box, ball_box)
            if bat_box is not None
            else None
        )

        result['distance'] = (
            round(distance, 2)
            if distance is not None
            else None
        )


        # =========================================================
        # REARM AFTER SEPARATION
        # =========================================================

        if (
            distance is not None
            and distance > self.rearm_distance
        ):
            self.rearmed = True


        # =========================================================
        # CALCULATE BAT-BALL BOX OVERLAP
        # =========================================================

        overlap_area = 0.0

        if bat_box is not None:

            intersection_x1 = max(
                bat_box[0],
                ball_box[0]
            )

            intersection_y1 = max(
                bat_box[1],
                ball_box[1]
            )

            intersection_x2 = min(
                bat_box[2],
                ball_box[2]
            )

            intersection_y2 = min(
                bat_box[3],
                ball_box[3]
            )

            intersection_width = max(
                0.0,
                intersection_x2 - intersection_x1
            )

            intersection_height = max(
                0.0,
                intersection_y2 - intersection_y1
            )

            overlap_area = (
                intersection_width
                * intersection_height
            )

        overlap = overlap_area > 0.0

        result['overlap'] = overlap
        result['overlap_area'] = round(
            overlap_area,
            2
        )


        # =========================================================
        # DIRECT OVERLAP VALIDATION
        # =========================================================
        #
        # IMPORTANT:
        #
        # Overlap is our strongest image-space contact evidence.
        #
        # However, do NOT accept a completely isolated overlap at
        # the beginning of a video with no prior ball evidence and
        # no active batting phase.
        #
        # This prevents the false impact that happened at frame 2
        # in the previous test video.
        #
        # Phase is NOT a hard requirement.
        #
        # A prior real ball observation is enough even when the
        # phase detector incorrectly reports Backlift/Unknown.
        # =========================================================

        has_prior_ball_evidence = len(self.history) > 0

        active_contact_phase = normalized_phase in (
            'downswing',
            'impact',
            'follow through'
        )

        overlap_ready = (
            has_prior_ball_evidence
            or active_contact_phase
        )


        # =========================================================
        # OVERLAP SEEN BUT NOT ENOUGH CONTEXT YET
        # =========================================================

        overlap_waiting_for_context = (
            overlap
            and not overlap_ready
        )

        if overlap_waiting_for_context:

            result['status'] = (
                'Overlap observed but waiting for prior '
                'ball evidence/contact phase'
            )

            # DO NOT return here.
            #
            # This real ball detection must continue below and be
            # stored in history so later frames have temporal
            # evidence.


        # =========================================================
        # DIRECT OVERLAP IMPACT
        # =========================================================

        elif overlap:

            # -----------------------------------------------------
            # Prevent repeated events during the same contact.
            # -----------------------------------------------------

            if not self.rearmed:

                result['status'] = (
                    'Bat-ball overlap continues; '
                    'waiting for separation'
                )

                return self._finish(result)


            # -----------------------------------------------------
            # Cooldown protects against duplicate impacts.
            # -----------------------------------------------------

            if self.cooldown_remaining > 0:

                result['status'] = (
                    'Bat-ball overlap detected during cooldown'
                )

                return self._finish(result)


            # -----------------------------------------------------
            # Fresh detection quality
            # -----------------------------------------------------

            try:
                bat_quality = float(
                    bat_detection.get(
                        'confidence',
                        0.0
                    )
                )
            except Exception:
                bat_quality = 0.0

            try:
                ball_quality = float(
                    ball.get(
                        'confidence',
                        0.0
                    )
                )
            except Exception:
                ball_quality = 0.0


            evidence_confidence = round(
                100.0
                * min(
                    bat_quality,
                    ball_quality
                ),
                1
            )


            # -----------------------------------------------------
            # REGISTER IMPACT
            # -----------------------------------------------------

            self.impact_count += 1

            self.last_impact_frame = frame_number

            self.rearmed = False

            self.impact_detected = True


            # Previous motion samples must not later produce
            # another delayed impact for this same contact.
            self.history.clear()


            self.cooldown_remaining = (
                self.cooldown_frames
            )


            result.update(

                detected=True,

                event=True,

                confidence=evidence_confidence,

                confidence_kind=(
                    'fresh detection overlap evidence'
                ),

                distance=0.0,

                impact_frame=frame_number,

                confirmation_frame=frame_number,

                impact_count=self.impact_count,

                cooldown=self.cooldown_remaining,

                overlap=True,

                overlap_area=round(
                    overlap_area,
                    2
                ),

                status=(
                    'Impact estimated from fresh '
                    'bat-ball bounding-box overlap'
                )
            )


            return self._finish(result)


        # =========================================================
        # BUILD CURRENT MOTION SAMPLE
        # =========================================================

        diameter = max(
            ball_box[2] - ball_box[0],
            ball_box[3] - ball_box[1]
        )

        sample = dict(

            frame=frame_number,

            ball=self._center(
                ball_box
            ),

            bat_box=bat_box,

            ball_box=ball_box,

            distance=distance,

            diameter=diameter,

            phase=normalized_phase,

            phase_eligible=phase_eligible,

            quality=float(
                ball['confidence']
            ),

            bat_quality=(
                float(
                    bat_detection['confidence']
                )
                if bat_box is not None
                else 0.0
            )
        )


        # =========================================================
        # BALL OUTLIER VALIDATION
        # =========================================================

        if self.history:

            previous = self.history[-1]

            dt = (
                frame_number
                - previous['frame']
            )

            jump = math.dist(
                previous['ball'],
                sample['ball']
            )

            size_ratio = (
                diameter
                / previous['diameter']
            )

            reason = None


            # -----------------------------------------------------
            # Ball box suddenly changed size
            # -----------------------------------------------------

            if not 0.5 <= size_ratio <= 2.0:

                reason = (
                    'ball size changed abruptly'
                )


            # -----------------------------------------------------
            # Ball jumped impossibly far
            # -----------------------------------------------------

            elif jump > (
                20
                * max(
                    diameter,
                    previous['diameter']
                )
                * dt
            ):

                reason = (
                    'ball position jumped abruptly'
                )


            # -----------------------------------------------------
            # Compare against recent velocity
            # -----------------------------------------------------

            elif len(self.history) >= 2:

                velocity = self._velocity(
                    self.history[-2],
                    previous
                )

                expected = tuple(
                    p + v * dt
                    for p, v in zip(
                        previous['ball'],
                        velocity
                    )
                )

                residual = math.dist(
                    expected,
                    sample['ball']
                )

                tolerance = max(

                    4 * max(
                        diameter,
                        previous['diameter']
                    ),

                    2
                    * math.hypot(
                        *velocity
                    )
                    * dt
                )

                if residual > tolerance:

                    reason = (
                        'ball position inconsistent '
                        'with recent observations'
                    )


            # -----------------------------------------------------
            # Reject outlier
            # -----------------------------------------------------

            if reason:

                result['ball_input'][
                    'accepted'
                ] = False

                result['ball_input'][
                    'reason'
                ] = reason

                result['status'] = (
                    'Ball outlier rejected; '
                    'recent observations retained'
                )

                return self._finish(result)


        # =========================================================
        # SAVE REAL BALL OBSERVATION
        # =========================================================

        self.history.append(sample)


        if overlap_waiting_for_context:

            result['status'] = (
                'Overlap observed; stored as first '
                'temporal ball evidence'
            )

        else:

            result['status'] = (
                'Collecting motion evidence'
            )


        # =========================================================
        # WAIT FOR FULL MOTION WINDOW
        # =========================================================

        if len(self.history) < self.history.maxlen:

            return self._finish(result)


        # =========================================================
        # EXISTING MOTION-BASED FALLBACK
        # =========================================================

        samples = list(
            self.history
        )

        n = self.confirmation_frames


        # =========================================================
        # CHECK FOR EXCESSIVE FRAME GAPS
        # =========================================================

        if (
            samples[-1]['frame']
            - samples[0]['frame']
            > 2 * n + 1
        ):

            result['status'] = (
                'Too many gaps in motion window'
            )

            return self._finish(result)


        # =========================================================
        # MIDDLE SAMPLE = MOTION CANDIDATE
        # =========================================================

        candidate = samples[n]

        result['candidate_frame'] = (
            candidate['frame']
        )

        result['candidate_phase'] = (
            candidate['phase']
        )


        # =========================================================
        # FRESH BAT REQUIRED FOR MOTION FALLBACK
        # =========================================================

        if candidate['bat_box'] is None:

            result['status'] = (
                'Candidate needs a fresh, valid bat '
                'detection; ball history retained'
            )

            return self._finish(result)


        # =========================================================
        # PHASE CHECK
        # =========================================================
        #
        # NOTE:
        # This phase requirement applies ONLY to the old
        # motion/proximity fallback.
        #
        # It does NOT apply to direct overlap contact.
        # =========================================================

        if not candidate['phase_eligible']:

            result['status'] = (
                'Candidate phase not eligible'
            )

            return self._finish(result)


        # =========================================================
        # PROXIMITY CHECK
        # =========================================================

        if (
            candidate['distance']
            > self.impact_distance
        ):

            result['status'] = (
                'Candidate too far from bat'
            )

            return self._finish(result)


        # =========================================================
        # CLOSEST APPROACH CHECK
        # =========================================================

        noise = max(
            1.0,
            candidate['diameter'] * 0.25
        )


        candidate_min_distance = min(

            self.box_distance(
                candidate['bat_box'],
                s['ball_box']
            )

            for s in samples
        )


        if (
            candidate['distance']
            > candidate_min_distance + noise
        ):

            result['status'] = (
                'Candidate not at closest approach'
            )

            return self._finish(result)


        # =========================================================
        # BALL VELOCITY BEFORE / AFTER CANDIDATE
        # =========================================================

        incoming = self._velocity(
            samples[0],
            candidate
        )

        outgoing = self._velocity(
            candidate,
            samples[-1]
        )


        speed_in = math.hypot(
            *incoming
        )

        speed_out = math.hypot(
            *outgoing
        )


        result['ball_speed'] = round(
            speed_out,
            2
        )


        # =========================================================
        # MINIMUM MOTION
        # =========================================================

        if (
            min(
                speed_in,
                speed_out
            )
            < self.min_ball_speed
        ):

            result['status'] = (
                'Ball motion too small to confirm'
            )

            return self._finish(result)


        # =========================================================
        # TRAJECTORY CONSISTENCY
        # =========================================================

        for group, velocity in (

            (
                samples[:n + 1],
                incoming
            ),

            (
                samples[n:],
                outgoing
            )

        ):

            if any(

                self._cos(
                    self._velocity(a, b),
                    velocity
                ) < 0.7

                for a, b
                in zip(
                    group,
                    group[1:]
                )
            ):

                result['status'] = (
                    'Inconsistent ball trajectory'
                )

                return self._finish(result)


        # =========================================================
        # DIRECTION / SPEED CHANGE
        # =========================================================

        angle = math.degrees(

            math.acos(
                self._cos(
                    incoming,
                    outgoing
                )
            )
        )


        speed_change = (

            abs(
                speed_out - speed_in
            )

            / max(
                speed_in,
                speed_out
            )
        )


        residual = (

            math.hypot(
                *(
                    b - a

                    for a, b
                    in zip(
                        incoming,
                        outgoing
                    )
                )
            )

            * n
        )


        result.update(

            direction_change=round(
                angle,
                1
            ),

            speed_change=round(
                speed_change,
                3
            )
        )


        # =========================================================
        # MOTION CHANGE REQUIREMENT
        # =========================================================

        if (

            residual
            < max(
                2.0,
                candidate['diameter']
            )

            or (

                angle
                < self.direction_change_degrees

                and

                speed_change
                < self.speed_change_ratio

            )
        ):

            result['status'] = (
                'Proximity only: no motion change'
            )

            return self._finish(result)


        # =========================================================
        # REARM / COOLDOWN CHECK
        # =========================================================

        if (
            not self.rearmed
            or self.cooldown_remaining
        ):

            result['status'] = (
                'Waiting for separation/cooldown'
            )

            return self._finish(result)


        # =========================================================
        # MOTION-BASED FALLBACK IMPACT
        # =========================================================

        self.impact_count += 1

        self.last_impact_frame = (
            candidate['frame']
        )

        self.rearmed = False

        self.history.clear()

        self.impact_detected = True


        self.cooldown_remaining = max(

            0,

            self.cooldown_frames
            - (
                frame_number
                - candidate['frame']
            )
        )


        result.update(

            detected=True,

            event=True,

            confidence=round(

                100
                * min(

                    candidate[
                        'bat_quality'
                    ],

                    min(
                        s['quality']
                        for s in samples
                    )
                ),

                1
            ),

            distance=round(
                candidate['distance'],
                2
            ),

            impact_frame=(
                candidate['frame']
            ),

            confirmation_frame=(
                frame_number
            ),

            impact_count=(
                self.impact_count
            ),

            cooldown=(
                self.cooldown_remaining
            ),

            status=(
                'Impact estimated from '
                'proximity and motion'
            )
        )


        return self._finish(result)

    @staticmethod
    def _input_summary(detection, validated_box, threshold):
        if not isinstance(detection, dict):
            return dict(accepted=False, reason='missing/detection not a dict', confidence=None, box=None)
        try:
            confidence = float(detection.get('confidence', 0))
            confidence = confidence if math.isfinite(confidence) else None
        except (TypeError, ValueError, OverflowError):
            confidence = None
        try:
            box = [float(v) for v in detection.get('box', ())]
            box = box if len(box) == 4 and all(math.isfinite(v) for v in box) else None
        except (TypeError, ValueError, OverflowError):
            box = None
        tracker = any(detection.get(k, False) for k in
                      ('predicted', 'is_predicted', 'tracked', 'tracker_only'))
        if validated_box is not None:
            reason = 'accepted'
        elif tracker:
            reason = 'tracked/predicted'
        elif box is None or box[2] <= box[0] or box[3] <= box[1]:
            reason = 'invalid/missing box'
        elif confidence is None or not 0 <= confidence <= 1:
            reason = 'invalid confidence'
        else:
            reason = f'confidence {confidence:.3f} below {threshold:.3f}'
        return dict(accepted=validated_box is not None, reason=reason,
                    confidence=confidence, box=box)

    def _finish(self, result):
        if self.debug:
            print(f"[IMPACT] frame={self.last_frame} candidate={result.get('candidate_frame')} "
                  f"gap={result['distance']} event={result['event']} status={result['status']} "
                  f"bat={result.get('bat_input')} ball={result.get('ball_input')}")
        return result
