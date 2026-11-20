from collections import deque
import math
import numpy as np


class AngleSmoother:
    """
    Smooths angle values using a moving average.
    """

    def __init__(self, window_size=5):

        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError("window_size must be a positive integer")

        self.window_size = window_size
        self.history = {}

    def smooth(self, angle_name, value):

        if value is None:
            return None

        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(value):
            return None

        if angle_name not in self.history:
            self.history[angle_name] = deque(
                maxlen=self.window_size
            )

        self.history[angle_name].append(value)

        return round(
            float(np.mean(self.history[angle_name])),
            2
        )

    def clear(self):
        """
        Reset all stored angle history.
        """
        self.history.clear()