"""Adaptive cursor smoothing.

MediaPipe fingertip coordinates jitter by a few pixels even when a hand is held
still, which turns a slow deliberate stroke into a wobbly line. A fixed low-pass
filter fixes the wobble but makes fast strokes lag behind the finger. This module
interpolates the filter coefficient against cursor speed: heavy smoothing when
the hand is nearly stationary, light smoothing when it is moving quickly.
"""

from __future__ import annotations

import math

from . import config


class CursorSmoother:
    """Speed-adaptive exponential moving average over cursor positions."""

    def __init__(
        self,
        alpha_slow: float = config.ALPHA_SLOW,
        alpha_fast: float = config.ALPHA_FAST,
        speed_scale: float = config.SMOOTHING_SPEED_SCALE,
    ) -> None:
        self.alpha_slow = alpha_slow
        self.alpha_fast = alpha_fast
        self.speed_scale = speed_scale
        self._x: float | None = None
        self._y: float | None = None
        self._last_raw: tuple[float, float] | None = None

    def reset(self) -> None:
        """Forget history so the next sample is taken verbatim."""
        self._x = self._y = None
        self._last_raw = None

    def alpha_for_speed(self, distance: float) -> float:
        """Blend factor for a given per-frame travel distance."""
        ratio = min(distance / self.speed_scale, 1.0)
        return self.alpha_slow + (self.alpha_fast - self.alpha_slow) * ratio

    def update(self, raw_x: float, raw_y: float) -> tuple[int, int]:
        """Feed a raw fingertip position, get the smoothed position back."""
        if self._x is None or self._y is None or self._last_raw is None:
            self._x, self._y = float(raw_x), float(raw_y)
            self._last_raw = (float(raw_x), float(raw_y))
            return int(raw_x), int(raw_y)

        distance = math.hypot(raw_x - self._last_raw[0], raw_y - self._last_raw[1])
        alpha = self.alpha_for_speed(distance)

        self._x += alpha * (raw_x - self._x)
        self._y += alpha * (raw_y - self._y)
        self._last_raw = (float(raw_x), float(raw_y))
        return int(self._x), int(self._y)
