"""Mutable application state, gathered into one object.

The reference implementation this project draws from kept ~30 module-level
globals and reached for `global` in a dozen functions. Collecting the same data
into `AppState` makes the flow of mutation explicit and lets the gesture handlers
be tested without a camera.
"""

from dataclasses import dataclass, field
from enum import Enum

from . import config


class AIState(Enum):
    """Lifecycle of an AI shape snap.

    IDLE      -> nothing pending
    ARMED     -> user tapped AI; waiting for them to draw a stroke
    WAITING   -> stroke captured; waiting for a fist to confirm
    ANIMATING -> snap accepted; crossfade in progress
    """

    IDLE = "idle"
    ARMED = "armed"
    WAITING = "waiting"
    ANIMATING = "animating"


Point = tuple[int, int]


@dataclass
class AppState:
    """Everything the main loop mutates frame to frame."""

    # Active tool / styling
    active_tool: str = "brush"
    draw_color: config.BGR = config.DEFAULT_COLOR
    brush_thickness: int = config.DEFAULT_BRUSH
    eraser_thickness: int = config.DEFAULT_ERASER

    # Stroke capture
    stroke_started: bool = False
    live_stroke: list[Point] = field(default_factory=list)
    current_stroke: list[Point] = field(default_factory=list)
    last_draw: Point | None = None

    # AI snap
    ai_state: AIState = AIState.IDLE
    pending_stroke: list[Point] = field(default_factory=list)
    ai_result: str = ""
    ai_result_timer: int = 0
    fist_count: int = 0

    # UI feedback
    settings_open: bool = False
    save_msg_timer: int = 0
    fps: float = 30.0
    hand_detected: bool = False

    @property
    def active_thickness(self) -> int:
        """Thickness of whichever tool is currently selected."""
        return (
            self.eraser_thickness if self.active_tool == "eraser" else self.brush_thickness
        )

    @property
    def is_erasing(self) -> bool:
        return self.active_tool == "eraser"

    def set_thickness(self, value: int) -> None:
        """Apply a resize to whichever tool is active."""
        if self.active_tool == "eraser":
            self.eraser_thickness = value
        else:
            self.brush_thickness = value

    def reset_stroke(self) -> None:
        """Abandon the in-progress stroke without committing it."""
        self.stroke_started = False
        self.last_draw = None
        self.live_stroke = []

    def announce(self, message: str, frames: int = config.AI_RESULT_FRAMES) -> None:
        """Show a transient status message in the results banner."""
        self.ai_result = message
        self.ai_result_timer = frames

    def tick_timers(self) -> None:
        """Advance the one-frame-per-tick countdowns."""
        if self.ai_result_timer > 0:
            self.ai_result_timer -= 1
        if self.save_msg_timer > 0:
            self.save_msg_timer -= 1
