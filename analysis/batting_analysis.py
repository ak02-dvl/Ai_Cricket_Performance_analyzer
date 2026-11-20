from analysis.angle_calculator import calculate_angle, get_point
from analysis.joint_config import JOINTS



def calculate_joint_angles(landmarks, image_shape=None):
    """
    Calculate 2D joint angles.

    image_shape=None:
        Legacy normalized-coordinate calculation.

    image_shape=(height, width, channels):
        Experimental pixel-scaled calculation.

    The legacy calculation remains the default.
    """

    if landmarks is None:
        return None

    # Optional geometry correction
    if image_shape is not None:
        if len(image_shape) < 2:
            raise ValueError("Invalid pose image shape")

        image_height, image_width = image_shape[:2]

        if image_height <= 0 or image_width <= 0:
            raise ValueError("Invalid pose image dimensions")

    angles = {}

    for joint_name, (a, b, c) in JOINTS.items():

        p1 = get_point(landmarks, a)
        p2 = get_point(landmarks, b)
        p3 = get_point(landmarks, c)

        if any(point is None for point in (p1, p2, p3)):
            continue

        # Experimental geometry correction.
        # Uses dimensions of the actual image
        # supplied to MediaPipe.
        if image_shape is not None:

            p1 = (
                p1[0] * image_width,
                p1[1] * image_height
            )

            p2 = (
                p2[0] * image_width,
                p2[1] * image_height
            )

            p3 = (
                p3[0] * image_width,
                p3[1] * image_height
            )

        angle = calculate_angle(p1, p2, p3)

        if angle is None:
            continue

        angles[joint_name] = round(angle, 2)

    return angles
