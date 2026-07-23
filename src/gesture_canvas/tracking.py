"""Hand tracking and finger-state extraction.

Evolved from the project's original `HandTracking_GestureRecognition_Module`:
the finger vector now covers all five digits (the original omitted the thumb),
thumb direction accounts for handedness, and the MediaPipe API is accessed
through a backend so the tracker works on both the legacy `solutions` API and
the newer Tasks API. See `backends.py` for that split.

Landmark drawing is done here with plain OpenCV rather than MediaPipe's drawing
helpers, since those only exist in the legacy API.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from . import config
from .backends import HandBackend, create_backend

Landmark = list[int]  # [id, x, y]
BBox = tuple[int, int, int, int]

#: Pairs of landmark ids forming the skeleton, for debug drawing.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                    # palm
)


class HandDetector:
    """Tracks one hand and answers the queries this application needs."""

    #: Landmark ids of the fingertips, thumb first.
    TIP_IDS: tuple[int, ...] = (4, 8, 12, 16, 20)

    #: Number of landmarks MediaPipe reports for a complete hand.
    LANDMARK_COUNT: int = 21

    def __init__(
        self,
        static_mode: bool = False,
        max_hands: int = 1,
        detection_confidence: float = config.DETECTION_CONFIDENCE,
        tracking_confidence: float = config.TRACKING_CONFIDENCE,
        model_path: Path | None = None,
        backend: HandBackend | None = None,
    ) -> None:
        self._backend = backend or create_backend(
            static_mode=static_mode,
            max_hands=max_hands,
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
            model_path=model_path,
        )

        self.lm_list: list[Landmark] = []
        self.bbox: BBox | None = None
        self.hand_type: str = "Unknown"
        self._raw_landmarks = None

    @property
    def backend_name(self) -> str:
        return getattr(self._backend, "name", "unknown")

    # ── Detection ────────────────────────────────────────────────────────────
    def find_hands(self, frame: np.ndarray, draw: bool = False) -> np.ndarray:
        """Run detection on a BGR frame; optionally annotate it in place."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._raw_landmarks, self.hand_type = self._backend.process(rgb)

        if draw and self._raw_landmarks:
            self._draw_skeleton(frame, self._raw_landmarks)
        return frame

    def find_position(
        self, frame: np.ndarray, hand_no: int = 0, draw: bool = False
    ) -> tuple[list[Landmark], BBox | None]:
        """Return pixel-space landmarks and the bounding box for the tracked hand."""
        self.lm_list = []
        self.bbox = None

        if not self._raw_landmarks:
            return self.lm_list, self.bbox

        height, width = frame.shape[:2]
        xs: list[int] = []
        ys: list[int] = []

        for lm_id, landmark in enumerate(self._raw_landmarks):
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            xs.append(cx)
            ys.append(cy)
            self.lm_list.append([lm_id, cx, cy])
            if draw:
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        self.bbox = (min(xs), min(ys), max(xs), max(ys))
        if draw:
            x1, y1, x2, y2 = self.bbox
            cv2.rectangle(frame, (x1 - 20, y1 - 20), (x2 + 20, y2 + 20), (0, 255, 0), 2)

        return self.lm_list, self.bbox

    @staticmethod
    def _draw_skeleton(frame: np.ndarray, landmarks) -> None:
        height, width = frame.shape[:2]
        points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (255, 255, 255), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 4, (255, 0, 255), cv2.FILLED)

    # ── Gesture queries ──────────────────────────────────────────────────────
    def fingers_up(self) -> list[int]:
        """Return ``[thumb, index, middle, ring, pinky]`` as 1 (up) or 0 (down)."""
        if len(self.lm_list) < self.LANDMARK_COUNT:
            return [0, 0, 0, 0, 0]

        # The thumb folds sideways, so compare x against the IP joint. The frame
        # is mirrored for a selfie view, which flips the comparison per hand.
        if self.hand_type == "Right":
            thumb_up = self.lm_list[4][1] < self.lm_list[3][1]
        else:
            thumb_up = self.lm_list[4][1] > self.lm_list[3][1]

        fingers = [1 if thumb_up else 0]
        # Remaining fingers curl downward: tip above the PIP joint means extended.
        for tip in self.TIP_IDS[1:]:
            fingers.append(1 if self.lm_list[tip][2] < self.lm_list[tip - 2][2] else 0)
        return fingers

    def find_distance(self, p1: int, p2: int) -> tuple[float, tuple[int, int, int, int, int, int]]:
        """Pixel distance between two landmarks, plus their endpoints and midpoint."""
        if len(self.lm_list) < self.LANDMARK_COUNT:
            return 0.0, (0, 0, 0, 0, 0, 0)

        x1, y1 = self.lm_list[p1][1], self.lm_list[p1][2]
        x2, y2 = self.lm_list[p2][1], self.lm_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return math.hypot(x2 - x1, y2 - y1), (x1, y1, x2, y2, cx, cy)

    def find_angle(self, p1: int, p2: int, p3: int) -> float:
        """Angle in degrees at ``p2`` formed by the points ``p1-p2-p3``."""
        if len(self.lm_list) < self.LANDMARK_COUNT:
            return 0.0

        x1, y1 = self.lm_list[p1][1], self.lm_list[p1][2]
        x2, y2 = self.lm_list[p2][1], self.lm_list[p2][2]
        x3, y3 = self.lm_list[p3][1], self.lm_list[p3][2]
        angle = math.degrees(
            math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
        )
        return abs(angle) % 360

    def close(self) -> None:
        """Release the underlying tracking graph."""
        self._backend.close()


def is_fist(fingers: list[int]) -> bool:
    """True when every finger is curled."""
    return all(value == 0 for value in fingers)
