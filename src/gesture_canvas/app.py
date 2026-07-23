"""GestureCanvas application: camera loop and gesture dispatch.

This module wires the pieces together and holds no drawing or geometry maths of
its own — that lives in `layers`, `shapes`, `fill`, `toolbar` and `overlay`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from . import config, overlay
from .fill import flood_fill
from .gestures import Gesture, classify
from .layers import LayerManager, SnapAnimation, UndoStack
from .shapes import classify_shape, draw_clean_shape, stroke_to_contour
from .smoothing import CursorSmoother
from .state import AIState, AppState
from .toolbar import Toolbar
from .tracking import HandDetector

logger = logging.getLogger(__name__)

#: Landmark ids used for pointing and pinching.
THUMB_TIP, INDEX_TIP, MIDDLE_TIP = 4, 8, 12

DEFAULT_SAVE_PATH = Path("saved_drawing.png")
DEFAULT_HEADER_ASSET = Path("assets/header.png")


class GestureCanvasApp:
    """Owns the capture device, the canvas, and the frame loop."""

    def __init__(
        self,
        camera_index: int = 0,
        header_asset: Path | None = DEFAULT_HEADER_ASSET,
        save_path: Path = DEFAULT_SAVE_PATH,
    ) -> None:
        self.camera_index = camera_index
        self.save_path = save_path

        self.state = AppState()
        self.layers = LayerManager(config.CAM_HEIGHT, config.CAM_WIDTH)
        self.undo = UndoStack()
        self.toolbar = Toolbar(header_asset)
        self.detector = HandDetector()
        self.smoother = CursorSmoother()
        self.animation = SnapAnimation()

        self._capture: cv2.VideoCapture | None = None
        self._resize_frames = 0
        self._stable_frames = 0
        self._last_pinch: float | None = None
        self._erasing = False
        self._last_frame_time = time.time()

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def run(self) -> None:
        """Open the camera and process frames until the user quits."""
        self._capture = cv2.VideoCapture(self.camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.camera_index}. "
                "Check that no other application is using it."
            )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)

        try:
            while True:
                success, frame = self._capture.read()
                if not success:
                    logger.warning("dropped frame from camera")
                    continue

                display = self.process_frame(frame)
                cv2.imshow("GestureCanvas", display)

                if not self._handle_key(cv2.waitKey(1) & 0xFF):
                    break
        finally:
            self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self.detector.close()
        cv2.destroyAllWindows()

    # ── Frame pipeline ───────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run one full frame: track, dispatch, composite, annotate."""
        # cap.set is only a request; resizing guarantees the hardcoded toolbar
        # zones line up with the image regardless of what the camera returns.
        frame = cv2.resize(frame, (config.CAM_WIDTH, config.CAM_HEIGHT))
        frame = cv2.flip(frame, 1)

        self._update_fps()
        self.toolbar.tick()
        self.state.tick_timers()

        self.detector.find_hands(frame, draw=config.SHOW_SKELETON)
        landmarks, _ = self.detector.find_position(frame)
        self.state.hand_detected = len(landmarks) >= HandDetector.LANDMARK_COUNT

        canvas = self._current_canvas()
        frame = overlay.key_canvas_over_feed(frame, canvas)

        cursor = self._dispatch(frame, landmarks)

        self._draw_overlays(frame, cursor)
        return frame

    def _current_canvas(self) -> np.ndarray:
        """Composite the layers, running the snap animation when one is active."""
        canvas = self.layers.composite()
        if not self.animation.active:
            return canvas

        canvas = self.animation.update(canvas)
        if self.animation.finished and self.state.ai_state is AIState.ANIMATING:
            self.layers.commit_preview(self.state.pending_stroke, self.state.brush_thickness)
            self.state.pending_stroke = []
            self.state.ai_state = AIState.IDLE
        return canvas

    def _dispatch(
        self, frame: np.ndarray, landmarks: list[list[int]]
    ) -> tuple[int, int] | None:
        """Route the current hand pose to its handler. Returns the toolbar cursor."""
        if not self.state.hand_detected:
            self._end_stroke()
            self._reset_resize()
            self.state.fist_count = 0
            self._erasing = False
            self.toolbar.hover.clear()
            return None

        index = (landmarks[INDEX_TIP][1], landmarks[INDEX_TIP][2])
        middle = (landmarks[MIDDLE_TIP][1], landmarks[MIDDLE_TIP][2])
        thumb = (landmarks[THUMB_TIP][1], landmarks[THUMB_TIP][2])
        gesture = classify(self.detector.fingers_up())

        self.toolbar.hover.update(*index)
        in_header = index[1] < config.HEADER_HEIGHT

        if gesture is not Gesture.RESIZE:
            self._reset_resize()
        if gesture is not Gesture.FIST:
            self.state.fist_count = 0
        if gesture is not Gesture.DRAW:
            self._end_stroke()
        if gesture is not Gesture.ERASE:
            self._erasing = False

        if gesture is Gesture.RESIZE:
            self._handle_resize(frame, thumb, index)
        elif gesture is Gesture.ERASE:
            self._handle_erase(frame, middle)
        elif gesture is Gesture.FILL:
            self._handle_fill(frame, index)
        elif gesture is Gesture.SELECT:
            self._handle_select(frame, index, middle, in_header)
        elif gesture is Gesture.FIST:
            self._handle_fist()
        elif gesture is Gesture.DRAW:
            self._handle_draw(frame, index)

        return index if in_header else None

    # ── Gesture handlers ─────────────────────────────────────────────────────
    def _handle_draw(self, frame: np.ndarray, index: tuple[int, int]) -> None:
        cv2.circle(frame, index, 10, self.state.draw_color, cv2.FILLED)
        if index[1] <= config.HEADER_HEIGHT:
            return

        point = self.smoother.update(*index)
        if not self.state.stroke_started:
            self.undo.push(self.layers.snapshot())
            self.state.stroke_started = True
            self.state.last_draw = point
            self.state.live_stroke = [point]

        if self.state.last_draw is not None:
            if self.state.is_erasing:
                self.layers.erase_stroke(self.state.last_draw, point,
                                         self.state.eraser_thickness)
            else:
                self.layers.draw_stroke(self.state.last_draw, point,
                                        self.state.draw_color, self.state.brush_thickness)

        self.state.live_stroke.append(point)
        self.state.last_draw = point

    def _handle_select(
        self,
        frame: np.ndarray,
        index: tuple[int, int],
        middle: tuple[int, int],
        in_header: bool,
    ) -> None:
        color = config.PANEL_CURSOR_COLOR if in_header else config.HIGHLIGHT_COLOR
        cv2.rectangle(
            frame,
            (min(index[0], middle[0]) - 10, min(index[1], middle[1]) - 10),
            (max(index[0], middle[0]) + 10, max(index[1], middle[1]) + 10),
            color, 2, cv2.LINE_AA,
        )
        if in_header:
            tool, swatch = self.toolbar.click(*index)
            if tool is not None:
                self._apply_selection(tool, swatch)

    def _handle_erase(self, frame: np.ndarray, middle: tuple[int, int]) -> None:
        """The original four-finger eraser: a circle stamped at the middle fingertip.

        Kept distinct from the toolbar eraser, which strokes between successive
        points. This one stamps a single disc per frame exactly as the original
        `Deploy.py` did, and does not disturb the selected tool.
        """
        radius = max(self.state.eraser_thickness // 2, 1)
        cv2.circle(frame, middle, radius, (235, 235, 235), 2, cv2.LINE_AA)
        if middle[1] <= config.HEADER_HEIGHT:
            return

        # One undo entry per continuous erase, matching how strokes behave. The
        # original pushed no history at all, which made erasing unrecoverable.
        if not self._erasing:
            self.undo.push(self.layers.snapshot())
            self._erasing = True

        cv2.circle(self.layers.base, middle, radius, (0, 0, 0), cv2.FILLED)

    def _handle_fill(self, frame: np.ndarray, index: tuple[int, int]) -> None:
        if index[1] <= config.HEADER_HEIGHT:
            return

        cv2.drawMarker(frame, index, self.state.draw_color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        cv2.circle(frame, index, 5, (235, 235, 235), 1, cv2.LINE_AA)
        if not self.toolbar.accepting_clicks:
            return

        snapshot = self.layers.snapshot()
        filled, result = flood_fill(self.layers.base, index[0], index[1], self.state.draw_color)
        if result.success:
            self.undo.push(snapshot)
            self.layers.base = filled
        if result.message:
            self.state.announce(result.message, frames=90)
            self.toolbar.start_cooldown(35)

    def _handle_fist(self) -> None:
        self.state.fist_count += 1
        if (
            self.state.fist_count == config.FIST_CONFIRM
            and self.state.ai_state is AIState.WAITING
        ):
            self._fire_snap()

    def _handle_resize(
        self, frame: np.ndarray, thumb: tuple[int, int], index: tuple[int, int]
    ) -> None:
        self._resize_frames += 1
        # Ignore the first few frames: fingers pass through this pose in transit
        # to other gestures, and acting immediately would resize by accident.
        if self._resize_frames < config.RESIZE_DEBOUNCE:
            return

        distance = float(np.hypot(index[0] - thumb[0], index[1] - thumb[1]))
        if self._last_pinch is not None and abs(distance - self._last_pinch) < 4:
            self._stable_frames += 1
        else:
            self._stable_frames = 0
        self._last_pinch = distance

        low, high = (
            (config.MIN_ERASER, config.MAX_ERASER)
            if self.state.is_erasing
            else (config.MIN_BRUSH, config.MAX_BRUSH)
        )
        size = int(np.interp(distance, [config.PINCH_MIN_DIST, config.PINCH_MAX_DIST],
                             [low, high]))
        self.state.set_thickness(size)

        overlay.draw_resize_indicator(
            frame, thumb, index, size, self.state.draw_color,
            locked=self._stable_frames > config.RESIZE_LOCK,
        )

    # ── Toolbar actions ──────────────────────────────────────────────────────
    def _apply_selection(self, tool: str, swatch: config.BGR | None) -> None:
        if tool == "colors" and swatch is not None:
            self.state.draw_color = swatch
            if self.state.is_erasing:
                self.state.active_tool = "brush"
            return

        if tool == "undo":
            self._undo()
        elif tool == "clear":
            self.undo.push(self.layers.snapshot())
            self.layers.clear()
            self.state.current_stroke = []
            self.state.ai_state = AIState.IDLE
        elif tool == "save":
            self._save()
        elif tool == "ai":
            self._toggle_ai()
        elif tool == "settings":
            self.state.settings_open = not self.state.settings_open
        elif tool in {"brush", "eraser"}:
            self.state.active_tool = tool
            self.state.ai_state = AIState.IDLE
            self.layers.cancel_preview()
            self.state.settings_open = False

    def _undo(self) -> None:
        snapshot = self.undo.pop()
        if snapshot is None:
            self.state.announce("Nothing to undo", frames=60)
            return
        self.layers.restore(snapshot)

    def _save(self) -> None:
        if cv2.imwrite(str(self.save_path), self.layers.composite()):
            self.state.save_msg_timer = config.SAVE_MSG_FRAMES
        else:
            self.state.announce(f"Could not write {self.save_path}", frames=120)

    def _toggle_ai(self) -> None:
        if self.state.ai_state is AIState.IDLE:
            # A stroke already drawn can be snapped straight away.
            self.state.ai_state = (
                AIState.WAITING
                if len(self.state.current_stroke) > 3
                else AIState.ARMED
            )
        else:
            self.state.ai_state = AIState.IDLE
            self.layers.cancel_preview()

    # ── AI snap ──────────────────────────────────────────────────────────────
    def _fire_snap(self) -> None:
        """Recognise the captured stroke and start the crossfade to a clean shape."""
        stroke = self.state.current_stroke
        if len(stroke) <= 3:
            self.state.announce("No stroke to snap - draw a shape first", frames=120)
            self.state.ai_state = AIState.IDLE
            return

        contour, approx = stroke_to_contour(stroke, self.layers.height, self.layers.width)
        if contour is None:
            self.state.announce("Stroke too small to recognise", frames=120)
            self.state.ai_state = AIState.IDLE
            return

        name, confidence = classify_shape(contour, approx)
        if name == "Freeform":
            self.state.announce("No recognisable shape - keeping your drawing", frames=120)
            self.state.ai_state = AIState.IDLE
            return

        before = self.layers.composite()
        self.undo.push(self.layers.snapshot())
        self.layers.start_preview(
            lambda target: draw_clean_shape(
                target, name, contour, approx,
                self.state.draw_color, self.state.brush_thickness,
            )
        )

        self.animation.start(before, self.layers.composite())
        self.state.pending_stroke = list(stroke)
        self.state.current_stroke = []
        self.state.ai_state = AIState.ANIMATING
        self.state.announce(f"Snapped: {name} | Confidence {confidence:.0%}")

    # ── Stroke bookkeeping ───────────────────────────────────────────────────
    def _end_stroke(self) -> None:
        """Commit the in-progress stroke as the candidate for AI snapping."""
        if self.state.stroke_started and len(self.state.live_stroke) > 2:
            self.state.current_stroke = list(self.state.live_stroke)
            if self.state.ai_state is AIState.ARMED:
                self.state.ai_state = AIState.WAITING
        self.state.reset_stroke()
        self.smoother.reset()

    def _reset_resize(self) -> None:
        self._resize_frames = 0
        self._stable_frames = 0
        self._last_pinch = None

    # ── Presentation ─────────────────────────────────────────────────────────
    def _draw_overlays(self, frame: np.ndarray, cursor: tuple[int, int] | None) -> None:
        badge = overlay.AI_BADGES.get(self.state.ai_state)
        self.toolbar.draw(
            frame,
            active_tool=self.state.active_tool,
            draw_color=self.state.draw_color,
            ai_label=badge,
            ai_color=overlay.AI_COLORS.get(self.state.ai_state),
            cursor=cursor,
        )

        if self.state.settings_open:
            overlay.draw_settings_panel(frame, self.state)
        if self.state.ai_state is not AIState.IDLE:
            overlay.draw_ai_hint(frame, self.state)
        if self.state.ai_result and self.state.ai_result_timer > 0:
            overlay.draw_status_banner(frame, self.state.ai_result)
        if self.state.save_msg_timer > 0:
            overlay.draw_save_confirmation(frame)

    def _update_fps(self) -> None:
        now = time.time()
        instant = 1.0 / max(now - self._last_frame_time, 1e-6)
        # Exponential average, so the readout does not flicker frame to frame.
        self.state.fps = 0.9 * self.state.fps + 0.1 * instant
        self._last_frame_time = now

    def _handle_key(self, key: int) -> bool:
        """Handle a keypress. Returns False to quit."""
        if key in (ord("q"), 27):
            return False
        if key == ord("z"):
            self._undo()
        elif key == ord("s"):
            self._save()
        elif key == ord("c"):
            self.undo.push(self.layers.snapshot())
            self.layers.clear()
        return True


def run(camera_index: int = 0) -> None:
    """Entry point used by ``main.py``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    GestureCanvasApp(camera_index=camera_index).run()
