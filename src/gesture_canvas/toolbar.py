"""The header toolbar: rendering, hit-testing, and hover/click stabilisation.

The toolbar is drawn from the same zone definitions used to hit-test it, so the
graphics can never drift out of alignment with the regions that respond to a
fingertip. An optional background image (``assets/header.png``) can be supplied
for styling; interactive elements are always drawn on top of it, so the app runs
identically with or without the asset.

Two stabilisers keep gesture input usable:

* **Hover debounce** — a zone must be held for several frames before it lights
  up, so tracking jitter near a boundary does not flicker between neighbours.
* **Click cooldown** — after a selection, input is ignored briefly. A fingertip
  lingers over a button for many frames, and every one of them would otherwise
  register as a fresh click.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config

#: Tools that toggle a mode rather than selecting a drawing implement.
_ACTION_TOOLS = frozenset({"undo", "clear", "save", "ai", "settings"})

#: Short labels drawn in each tool zone.
_TOOL_LABELS: dict[str, str] = {
    "brush": "BRUSH",
    "eraser": "ERASER",
    "undo": "UNDO",
    "clear": "CLEAR",
    "save": "SAVE",
    "ai": "AI",
    "settings": "SETUP",
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def hit_test(x: int, y: int) -> tuple[str | None, config.BGR | None]:
    """Resolve a point to ``(tool, color)``.

    Returns ``(None, None)`` below the header or in a gap between zones. A colour
    swatch resolves to ``("colors", swatch_color)``; the gaps between swatches
    resolve to ``("colors", None)`` so a near-miss does not change the colour.
    """
    if y >= config.HEADER_HEIGHT or not (0 <= x < config.CAM_WIDTH):
        return None, None

    for tool, (start, end) in config.TOOL_ZONES.items():
        if not start <= x < end:
            continue
        if tool != "colors":
            return tool, None
        for (swatch_start, swatch_end), color in config.COLOR_ZONES:
            if swatch_start <= x < swatch_end:
                return "colors", color
        return "colors", None
    return None, None


class HoverTracker:
    """Debounces which zone the fingertip is considered to be over."""

    def __init__(self, debounce: int = config.HOVER_DEBOUNCE) -> None:
        self.debounce = debounce
        self._raw: tuple[str | None, config.BGR | None] = (None, None)
        self._frames = 0
        self.tool: str | None = None
        self.color: config.BGR | None = None

    def update(self, x: int, y: int) -> None:
        candidate = hit_test(x, y)
        if candidate == self._raw:
            self._frames += 1
        else:
            self._raw = candidate
            self._frames = 0

        if self._frames >= self.debounce:
            self.tool, self.color = candidate

    def clear(self) -> None:
        self._raw = (None, None)
        self._frames = 0
        self.tool = None
        self.color = None


class Toolbar:
    """Renders the header and reports what the user selected."""

    def __init__(self, background_path: Path | str | None = None) -> None:
        self.background = self._load_background(background_path)
        self.hover = HoverTracker()
        self._cooldown = 0

    @staticmethod
    def _load_background(path: Path | str | None) -> np.ndarray | None:
        """Load an optional styling image; missing or unreadable is not an error."""
        if path is None:
            return None
        candidate = Path(path)
        if not candidate.is_file():
            return None
        image = cv2.imread(str(candidate))
        if image is None:
            return None
        return cv2.resize(image, (config.CAM_WIDTH, config.HEADER_HEIGHT))

    # ── Input ────────────────────────────────────────────────────────────────
    def tick(self) -> None:
        """Advance the click cooldown by one frame."""
        if self._cooldown > 0:
            self._cooldown -= 1

    @property
    def accepting_clicks(self) -> bool:
        return self._cooldown == 0

    def click(self, x: int, y: int) -> tuple[str | None, config.BGR | None]:
        """Register a selection, or return ``(None, None)`` if still cooling down."""
        if self._cooldown > 0:
            return None, None
        tool, color = hit_test(x, y)
        if tool is None:
            return None, None
        if tool == "colors" and color is None:
            return None, None
        self._cooldown = config.CLICK_COOLDOWN
        return tool, color

    def start_cooldown(self, frames: int = config.CLICK_COOLDOWN) -> None:
        """Suppress input for a while — used after a fill, which is not a click."""
        self._cooldown = max(self._cooldown, frames)

    # ── Rendering ────────────────────────────────────────────────────────────
    def draw(
        self,
        frame: np.ndarray,
        active_tool: str,
        draw_color: config.BGR,
        ai_label: str | None = None,
        ai_color: config.BGR | None = None,
        cursor: tuple[int, int] | None = None,
    ) -> None:
        """Draw the toolbar over the top of ``frame``."""
        self._draw_background(frame)
        self._draw_swatches(frame, draw_color)
        self._draw_tool_labels(frame, active_tool)
        self._draw_active_outline(frame, active_tool)
        self._draw_hover_outline(frame, active_tool)

        if ai_label and ai_color:
            self._draw_ai_badge(frame, ai_label, ai_color)
        if cursor is not None:
            cx, cy = cursor
            cv2.rectangle(frame, (cx - 18, cy - 18), (cx + 18, cy + 18),
                          config.PANEL_CURSOR_COLOR, 2, cv2.LINE_AA)

    def _draw_background(self, frame: np.ndarray) -> None:
        if self.background is not None:
            frame[0 : config.HEADER_HEIGHT, 0 : config.CAM_WIDTH] = self.background
            return
        frame[0 : config.HEADER_HEIGHT, 0 : config.CAM_WIDTH] = config.HEADER_BG
        # Sliced rather than drawn with cv2.line: a stroked line is centred on
        # its coordinate and would spill a row onto the canvas below.
        frame[config.HEADER_HEIGHT - 2 : config.HEADER_HEIGHT, :] = config.HEADER_DIVIDER
        for start, _ in list(config.TOOL_ZONES.values())[1:]:
            cv2.line(frame, (start, 12), (start, config.HEADER_HEIGHT - 12),
                     config.HEADER_DIVIDER, 1)

    def _draw_swatches(self, frame: np.ndarray, draw_color: config.BGR) -> None:
        for (start, end), color in config.COLOR_ZONES:
            cv2.rectangle(frame, (start, config.SWATCH_Y_START), (end, config.SWATCH_Y_END),
                          color, cv2.FILLED)
            if color == draw_color:
                cv2.rectangle(frame, (start - 3, config.SWATCH_Y_START - 3),
                              (end + 3, config.SWATCH_Y_END + 3),
                              config.HEADER_TEXT_ACTIVE, 2, cv2.LINE_AA)
            elif self.hover.color == color:
                cv2.rectangle(frame, (start - 3, config.SWATCH_Y_START - 3),
                              (end + 3, config.SWATCH_Y_END + 3),
                              config.HOVER_COLOR, 2, cv2.LINE_AA)

        start, _ = config.TOOL_ZONES["colors"]
        cv2.putText(frame, "COLORS", (start + 14, config.HEADER_HEIGHT - 14),
                    _FONT, 0.38, config.HEADER_TEXT, 1, cv2.LINE_AA)

    def _draw_tool_labels(self, frame: np.ndarray, active_tool: str) -> None:
        for tool, (start, end) in config.TOOL_ZONES.items():
            label = _TOOL_LABELS.get(tool)
            if label is None:
                continue
            color = config.HEADER_TEXT_ACTIVE if tool == active_tool else config.HEADER_TEXT
            (text_w, _), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
            x = start + (end - start - text_w) // 2
            cv2.putText(frame, label, (x, config.HEADER_HEIGHT // 2 + 6),
                        _FONT, 0.5, color, 1, cv2.LINE_AA)

    def _draw_active_outline(self, frame: np.ndarray, active_tool: str) -> None:
        if active_tool not in config.TOOL_ZONES:
            return
        start, end = config.TOOL_ZONES[active_tool]
        cv2.rectangle(frame, (start + 2, 4), (end - 2, config.HEADER_HEIGHT - 4),
                      config.HIGHLIGHT_COLOR, 3, cv2.LINE_AA)

    def _draw_hover_outline(self, frame: np.ndarray, active_tool: str) -> None:
        tool = self.hover.tool
        if tool is None or tool == active_tool or tool == "colors":
            return
        start, end = config.TOOL_ZONES[tool]
        cv2.rectangle(frame, (start + 2, 4), (end - 2, config.HEADER_HEIGHT - 4),
                      config.HOVER_COLOR, 2, cv2.LINE_AA)

    def _draw_ai_badge(self, frame: np.ndarray, label: str, color: config.BGR) -> None:
        start, end = config.TOOL_ZONES["ai"]
        cv2.rectangle(frame, (start + 2, 4), (end - 2, config.HEADER_HEIGHT - 4),
                      color, 3, cv2.LINE_AA)
        cv2.putText(frame, label, (start + 8, config.HEADER_HEIGHT - 10),
                    _FONT, 0.36, color, 1, cv2.LINE_AA)
