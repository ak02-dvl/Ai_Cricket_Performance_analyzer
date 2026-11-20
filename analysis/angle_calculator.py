import math
import numpy as np


def calculate_angle(point1, point2, point3):
    """
    Calculate the angle in degrees formed by:
        point1 -> point2 -> point3

    point2 is the vertex.
    """

    if (
        point1 is None
        or point2 is None
        or point3 is None
    ):
        return None

    try:

        p1 = np.array(point1, dtype=float)
        p2 = np.array(point2, dtype=float)
        p3 = np.array(point3, dtype=float)

    except (TypeError, ValueError):

        return None

    if (
        p1.shape != (2,)
        or p2.shape != (2,)
        or p3.shape != (2,)
    ):
        return None

    if not (
        np.isfinite(p1).all()
        and np.isfinite(p2).all()
        and np.isfinite(p3).all()
    ):
        return None

    vector1 = p1 - p2
    vector2 = p3 - p2

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return None

    cosine_value = np.dot(
        vector1,
        vector2
    ) / (norm1 * norm2)

    cosine_value = np.clip(
        cosine_value,
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(cosine_value)
    )

    return round(
        float(angle),
        2
    )

def get_point(landmarks, name):

    if not isinstance(landmarks, dict):
        return None

    lm = landmarks.get(name)

    if not isinstance(lm, dict):
        return None

    if "x" not in lm or "y" not in lm:
        return None

    try:

        x = float(lm["x"])
        y = float(lm["y"])

    except (TypeError, ValueError):

        return None

    if not (
        math.isfinite(x)
        and math.isfinite(y)
    ):
        return None

    return (
        x,
        y
    )