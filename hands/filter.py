"""One-Euro Filter implementation for 2D and 3D landmark smoothing.

Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
for Noisy Input in Human-Computer Interfaces", CHI 2012.
"""

import math
from typing import List, Optional, Tuple
import numpy as np


class LowPassFilter:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.last_value: Optional[float] = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if self.last_value is None:
            s = value
        else:
            s = self.alpha * value + (1.0 - self.alpha) * self.last_value
        self.last_value = s
        return s

    def reset(self):
        self.last_value = None


class OneEuroFilter:
    def __init__(
        self,
        t0: float,
        x0: float,
        min_cutoff: float = 1.0,
        beta: float = 0.01,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.last_time = float(t0)
        self.x_filt.filter(x0)
        self.dx_filt.filter(0.0)

    @staticmethod
    def _alpha(rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / max(1e-4, rate)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, t: float) -> float:
        dt = t - self.last_time
        if dt <= 0.0:
            return self.x_filt.last_value if self.x_filt.last_value is not None else x

        rate = 1.0 / dt
        self.last_time = t

        # Filter the derivative
        prev_x = self.x_filt.last_value
        dx = (x - prev_x) * rate if prev_x is not None else 0.0
        edx = self.dx_filt.filter(dx, self._alpha(rate, self.d_cutoff))

        # Dynamic cutoff frequency based on velocity magnitude
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filt.filter(x, self._alpha(rate, cutoff))

    def reset(self):
        self.x_filt.reset()
        self.dx_filt.reset()
        self.last_time = 0.0


class KeypointSmoother:
    """Manages 1-Euro filters for all 21 2D and 3D hand keypoints."""
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.01):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.filters_2d: List[Tuple[OneEuroFilter, OneEuroFilter]] = []
        self.filters_3d: List[Tuple[OneEuroFilter, OneEuroFilter, OneEuroFilter]] = []
        self.initialized = False

    def init_landmarks(self, kp2d: List[List[float]], kp3d: List[List[float]], t: float):
        self.filters_2d = []
        self.filters_3d = []
        for pt in kp2d:
            fx = OneEuroFilter(t, pt[0], self.min_cutoff, self.beta)
            fy = OneEuroFilter(t, pt[1], self.min_cutoff, self.beta)
            self.filters_2d.append((fx, fy))

        for pt in kp3d:
            fx = OneEuroFilter(t, pt[0], self.min_cutoff, self.beta)
            fy = OneEuroFilter(t, pt[1], self.min_cutoff, self.beta)
            fz = OneEuroFilter(t, pt[2], self.min_cutoff, self.beta)
            self.filters_3d.append((fx, fy, fz))

        self.initialized = True

    def smooth(
        self, kp2d: List[List[float]], kp3d: List[List[float]], t: float
    ) -> Tuple[List[List[float]], List[List[float]]]:
        if not self.initialized:
            self.init_landmarks(kp2d, kp3d, t)
            return kp2d, kp3d

        smooth_2d: List[List[float]] = []
        for idx, pt in enumerate(kp2d):
            fx, fy = self.filters_2d[idx]
            smooth_2d.append([fx.filter(pt[0], t), fy.filter(pt[1], t)])

        smooth_3d: List[List[float]] = []
        for idx, pt in enumerate(kp3d):
            fx, fy, fz = self.filters_3d[idx]
            smooth_3d.append([fx.filter(pt[0], t), fy.filter(pt[1], t), fz.filter(pt[2], t)])

        return smooth_2d, smooth_3d

