"""Central configuration for GestureCanvas.

Every magic number lives here. Toolbar hit-zones and the toolbar *rendering* both
derive from these constants, so the graphics can never drift out of sync with the
regions that respond to a fingertip.
"""

from dataclasses import dataclass
from typing import Final

BGR = tuple[int, int, int]

# ── Canvas ───────────────────────────────────────────────────────────────────
CAM_WIDTH: Final[int] = 1280
CAM_HEIGHT: Final[int] = 720
HEADER_HEIGHT: Final[int] = 100

# ── Brush / eraser ranges ────────────────────────────────────────────────────
MIN_BRUSH: Final[int] = 4
MAX_BRUSH: Final[int] = 50
MIN_ERASER: Final[int] = 20
MAX_ERASER: Final[int] = 120
DEFAULT_BRUSH: Final[int] = 15
DEFAULT_ERASER: Final[int] = 70

# Pinch distance (px) mapped onto the size ranges above.
PINCH_MIN_DIST: Final[int] = 20
PINCH_MAX_DIST: Final[int] = 200

# ── Gesture stability ────────────────────────────────────────────────────────
HOVER_DEBOUNCE: Final[int] = 5  # frames a hover must hold before it registers
CLICK_COOLDOWN: Final[int] = 20  # frames locked out after a toolbar click
FIST_CONFIRM: Final[int] = 8  # frames a fist must hold to fire an AI snap
RESIZE_DEBOUNCE: Final[int] = 12  # frames before a pinch starts resizing
RESIZE_LOCK: Final[int] = 20  # stable frames before the size reads "LOCKED"

# Adaptive cursor smoothing: slow motion uses ALPHA_SLOW (heavy smoothing),
# fast motion ramps toward ALPHA_FAST (responsive) to avoid lag on quick strokes.
ALPHA_SLOW: Final[float] = 0.18
ALPHA_FAST: Final[float] = 0.55
SMOOTHING_SPEED_SCALE: Final[float] = 40.0

# ── Undo / feedback ──────────────────────────────────────────────────────────
UNDO_DEPTH: Final[int] = 25
AI_RESULT_FRAMES: Final[int] = 200
SAVE_MSG_FRAMES: Final[int] = 90
SNAP_ANIM_MS: Final[int] = 220

# ── Palette ──────────────────────────────────────────────────────────────────
HIGHLIGHT_COLOR: Final[BGR] = (80, 60, 180)
HOVER_COLOR: Final[BGR] = (70, 70, 70)
PANEL_CURSOR_COLOR: Final[BGR] = (40, 40, 40)
HEADER_BG: Final[BGR] = (24, 24, 26)
HEADER_TEXT: Final[BGR] = (180, 180, 185)
HEADER_TEXT_ACTIVE: Final[BGR] = (245, 245, 245)
HEADER_DIVIDER: Final[BGR] = (48, 48, 52)
SUCCESS_COLOR: Final[BGR] = (0, 210, 85)

# Composite threshold: canvas pixels brighter than this are treated as ink when
# keying the canvas over the camera feed. Ink is never pure black by design.
INK_THRESHOLD: Final[int] = 10

DEFAULT_COLOR: Final[BGR] = (0, 0, 200)

# ── Toolbar layout ───────────────────────────────────────────────────────────
# Zones are expressed as fractions of CAM_WIDTH so the toolbar scales cleanly if
# the capture resolution changes.
_TOOL_SPANS: Final[tuple[tuple[str, float, float], ...]] = (
    ("colors", 0.000, 0.293),
    ("brush", 0.293, 0.394),
    ("eraser", 0.394, 0.494),
    ("undo", 0.494, 0.591),
    ("clear", 0.591, 0.688),
    ("save", 0.688, 0.788),
    ("ai", 0.788, 0.885),
    ("settings", 0.885, 1.000),
)

TOOL_ZONES: Final[dict[str, tuple[int, int]]] = {
    name: (int(start * CAM_WIDTH), int(end * CAM_WIDTH))
    for name, start, end in _TOOL_SPANS
}

PALETTE: Final[tuple[BGR, ...]] = (
    (0, 0, 200),  # red
    (0, 140, 255),  # orange
    (0, 200, 0),  # green
    (200, 0, 0),  # blue
    (120, 0, 120),  # purple
    (30, 30, 30),  # near-black
    (220, 220, 220),  # white
)

# Swatch geometry inside the "colors" zone.
SWATCH_Y_START: Final[int] = 26
SWATCH_Y_END: Final[int] = 74
_SWATCH_PAD: Final[int] = 14


def _build_color_zones() -> tuple[tuple[tuple[int, int], BGR], ...]:
    """Lay the palette out evenly across the colors zone."""
    zone_start, zone_end = TOOL_ZONES["colors"]
    usable = zone_end - zone_start - 2 * _SWATCH_PAD
    step = usable / len(PALETTE)
    width = int(step * 0.72)
    zones = []
    for index, color in enumerate(PALETTE):
        x1 = int(zone_start + _SWATCH_PAD + index * step)
        zones.append(((x1, x1 + width), color))
    return tuple(zones)


COLOR_ZONES: Final[tuple[tuple[tuple[int, int], BGR], ...]] = _build_color_zones()


@dataclass(frozen=True)
class ShapeConfig:
    """Tunables for the shape recognition pipeline."""

    resample_spacing: int = 6
    smooth_window: int = 5
    min_contour_area: int = 300
    approx_epsilon_ratio: float = 0.025
    confidence_threshold: float = 62.0
    min_stroke_points: int = 4
    #: Minimum overlap between the proposed ideal shape and the drawn stroke.
    #: Below this the proposal is rejected as not resembling what was drawn.
    min_fit_iou: float = 0.72


SHAPE_CONFIG: Final[ShapeConfig] = ShapeConfig()
