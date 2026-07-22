"""On-screen feedback drawn over the camera feed.

Kept separate from the main loop so the loop reads as gesture dispatch rather
than drawing code.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import config
from .state import AIState, AppState

_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Badge colour per AI phase, so the state is readable at a glance.
AI_COLORS: dict[AIState, config.BGR] = {
    AIState.ARMED: (0, 170, 70),
    AIState.WAITING: (0, 200, 255),
    AIState.ANIMATING: (200, 150, 0),
}

AI_BADGES: dict[AIState, str] = {
    AIState.ARMED: "DRAW",
    AIState.WAITING: "FIST",
    AIState.ANIMATING: "...",
}

AI_HINTS: dict[AIState, str] = {
    AIState.ARMED: "AI ready - draw a shape, then lower your finger",
    AIState.WAITING: "Shape captured - make a FIST to snap, or tap AI to cancel",
    AIState.ANIMATING: "Snapping...",
}


def _translucent_box(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    opacity: float = 0.85,
) -> None:
    """Darken a region so text stays legible over a busy camera image."""
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)


def draw_ai_hint(frame: np.ndarray, state: AppState) -> None:
    """Explain what the AI is waiting for, plus the fist-hold progress bar."""
    text = AI_HINTS.get(state.ai_state)
    if not text:
        return

    color = AI_COLORS.get(state.ai_state, (200, 200, 200))
    width = 560
    x = config.CAM_WIDTH // 2 - width // 2
    y = config.HEADER_HEIGHT + 10

    _translucent_box(frame, (x, y), (x + width, y + 34))
    cv2.rectangle(frame, (x, y), (x + width, y + 34), color, 1)
    cv2.putText(frame, text, (x + 12, y + 23), _FONT, 0.45, color, 1, cv2.LINE_AA)

    if state.ai_state is AIState.WAITING and state.fist_count > 0:
        progress = min(state.fist_count / config.FIST_CONFIRM, 1.0)
        cv2.rectangle(frame, (x, y + 34), (x + width, y + 39), (35, 35, 35), cv2.FILLED)
        cv2.rectangle(frame, (x, y + 34), (x + int(width * progress), y + 39),
                      color, cv2.FILLED)


def draw_settings_panel(frame: np.ndarray, state: AppState) -> None:
    """Live readout of tool, size, frame rate and tracking status."""
    x, _ = config.TOOL_ZONES["settings"]
    y = config.HEADER_HEIGHT + 4
    width = config.CAM_WIDTH - x
    height = 112

    _translucent_box(frame, (x, y), (x + width, y + height), opacity=0.9)
    cv2.rectangle(frame, (x, y), (x + width, y + height), config.HOVER_COLOR, 1)

    lines = [
        f"Tool : {state.active_tool.capitalize()}",
        f"Size : {state.active_thickness}px",
        f"FPS  : {state.fps:.0f}",
        f"Hand : {'yes' if state.hand_detected else 'no'}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (x + 12, y + 24 + index * 22),
                    _FONT, 0.46, (170, 170, 170), 1, cv2.LINE_AA)

    cv2.circle(frame, (x + width - 16, y + 16), 8, state.draw_color, cv2.FILLED)
    cv2.circle(frame, (x + width - 16, y + 16), 8, (90, 90, 90), 1)


def draw_status_banner(frame: np.ndarray, message: str) -> None:
    """Bottom banner for the last AI or fill result."""
    x1, y1 = 40, config.CAM_HEIGHT - 108
    x2, y2 = config.CAM_WIDTH - 40, config.CAM_HEIGHT - 14

    _translucent_box(frame, (x1, y1), (x2, y2), opacity=0.84)

    parts = message.split("|")
    cv2.putText(frame, parts[0].strip(), (x1 + 16, y1 + 40),
                _FONT, 0.9, config.SUCCESS_COLOR, 2, cv2.LINE_AA)
    if len(parts) > 1:
        cv2.putText(frame, parts[1].strip(), (x1 + 16, y1 + 70),
                    _FONT, 0.55, (120, 190, 120), 1, cv2.LINE_AA)


def draw_save_confirmation(frame: np.ndarray) -> None:
    cv2.putText(frame, "SAVED", (config.CAM_WIDTH // 2 - 90, config.CAM_HEIGHT // 2),
                _FONT, 2.0, config.SUCCESS_COLOR, 4, cv2.LINE_AA)


def draw_resize_indicator(
    frame: np.ndarray,
    thumb: tuple[int, int],
    index: tuple[int, int],
    size: int,
    color: config.BGR,
    locked: bool,
) -> None:
    """Preview the brush at its new size, between the pinching fingers."""
    cx, cy = (thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2
    radius = max(size // 2, 2)

    cv2.line(frame, thumb, index, config.HIGHLIGHT_COLOR, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius, color, cv2.FILLED)
    cv2.circle(frame, (cx, cy), radius, (200, 200, 200), 2, cv2.LINE_AA)

    label = "LOCKED" if locked else f"{size}px"
    cv2.putText(frame, label, (cx - 28, cy - radius - 14),
                _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def key_canvas_over_feed(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    """Composite the drawing onto the camera image.

    Ink is opaque and the rest of the canvas is transparent. Masking and OR-ing
    keeps the camera visible everywhere the user has not drawn.
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, holes = cv2.threshold(gray, config.INK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    holes = cv2.cvtColor(holes, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_or(cv2.bitwise_and(frame, holes), canvas)
