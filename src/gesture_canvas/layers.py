"""Layered canvas rendering, undo history, and the snap crossfade.

Two layers are kept:

* ``base``    - committed ink. Everything the user has actually drawn.
* ``preview`` - the AI's proposed clean shape, rendered but not committed. It can
  be discarded without touching ``base``, which is what makes the snap
  non-destructive until the user confirms it.

Black (0, 0, 0) is treated as "no ink" throughout; the composite is keyed over
the camera feed on that basis, so erasing is just drawing in black.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from . import config


class LayerManager:
    """Owns the drawing surfaces and how they combine."""

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.base: np.ndarray = np.zeros((height, width, 3), np.uint8)
        self.preview: np.ndarray | None = None

    # ── Compositing ──────────────────────────────────────────────────────────
    def composite(self) -> np.ndarray:
        """Flatten base + preview into a single image."""
        if self.preview is None:
            return self.base.copy()

        gray = cv2.cvtColor(self.preview, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, config.INK_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return np.where(mask3 > 0, self.preview, self.base)

    # ── Freehand drawing ─────────────────────────────────────────────────────
    def draw_stroke(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        color: config.BGR,
        thickness: int,
    ) -> None:
        """Draw a round-capped segment by stamping overlapping filled circles.

        Stamping rather than `cv2.line` keeps the stroke width uniform through
        direction changes and avoids the mitred corners a polyline would produce.
        """
        radius = max(thickness // 2, 1)
        span = math.hypot(end[0] - start[0], end[1] - start[1])
        steps = max(1, int(span / 3))
        for i in range(steps + 1):
            t = i / steps
            x = int(start[0] + (end[0] - start[0]) * t)
            y = int(start[1] + (end[1] - start[1]) * t)
            cv2.circle(self.base, (x, y), radius, color, cv2.FILLED)

    def erase_stroke(
        self, start: tuple[int, int], end: tuple[int, int], thickness: int
    ) -> None:
        """Erasing is drawing in the transparent colour."""
        self.draw_stroke(start, end, (0, 0, 0), thickness)

    # ── Snap lifecycle ───────────────────────────────────────────────────────
    def start_preview(self, render_fn) -> None:
        """Allocate a preview layer and let ``render_fn`` draw the clean shape."""
        self.preview = np.zeros((self.height, self.width, 3), np.uint8)
        render_fn(self.preview)

    def commit_preview(self, raw_stroke: list[tuple[int, int]], thickness: int) -> None:
        """Erase the messy source stroke, then bake the preview into the base."""
        if self.preview is None:
            return

        if len(raw_stroke) >= 2:
            # Wipe a band wider than the stroke so anti-aliased edges go too.
            wipe = np.zeros((self.height, self.width), np.uint8)
            points = np.array(raw_stroke, dtype=np.int32)
            cv2.polylines(wipe, [points], False, 255, thickness + 16, cv2.LINE_AA)
            self.base[wipe > 0] = (0, 0, 0)

        gray = cv2.cvtColor(self.preview, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, config.INK_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self.base = np.where(mask3 > 0, self.preview, self.base)
        self.preview = None

    def cancel_preview(self) -> None:
        self.preview = None

    # ── Whole-canvas operations ──────────────────────────────────────────────
    def clear(self) -> None:
        self.base = np.zeros((self.height, self.width, 3), np.uint8)
        self.preview = None

    def snapshot(self) -> np.ndarray:
        return self.base.copy()

    def restore(self, snapshot: np.ndarray) -> None:
        self.base = snapshot
        self.preview = None


class UndoStack:
    """Bounded stack of canvas snapshots."""

    def __init__(self, depth: int = config.UNDO_DEPTH) -> None:
        self.depth = depth
        self._stack: list[np.ndarray] = []

    def push(self, snapshot: np.ndarray) -> None:
        self._stack.append(snapshot)
        if len(self._stack) > self.depth:
            self._stack.pop(0)

    def pop(self) -> np.ndarray | None:
        return self._stack.pop() if self._stack else None

    def __len__(self) -> int:
        return len(self._stack)


class SnapAnimation:
    """Time-based crossfade between two composited frames."""

    def __init__(self, duration_ms: int = config.SNAP_ANIM_MS) -> None:
        self.duration = duration_ms / 1000.0
        self.active = False
        self._before: np.ndarray | None = None
        self._after: np.ndarray | None = None
        self._start_time = 0.0

    def start(self, before: np.ndarray, after: np.ndarray) -> None:
        self._before = before.copy()
        self._after = after.copy()
        self._start_time = time.time()
        self.active = True

    def update(self, fallback: np.ndarray) -> np.ndarray:
        """Return the current blend, or ``fallback`` when not animating."""
        if not self.active or self._before is None or self._after is None:
            return fallback

        progress = min((time.time() - self._start_time) / self.duration, 1.0)
        if progress >= 1.0:
            self.active = False
            return self._after.copy()

        # Smoothstep easing so the shape settles rather than snapping linearly.
        eased = progress * progress * (3 - 2 * progress)
        return cv2.addWeighted(self._before, 1 - eased, self._after, eased, 0)

    @property
    def finished(self) -> bool:
        return not self.active
