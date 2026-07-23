"""Integration tests for gesture dispatch.

The camera and MediaPipe are stubbed out, so these exercise the real state
machine and canvas without hardware. `process_frame` is not called: it invokes
MediaPipe and `cv2.imshow`. Instead `_dispatch` is driven directly with landmark
lists, which is where the gesture logic actually lives.
"""

from __future__ import annotations

import numpy as np
import pytest

from gesture_canvas import config
from gesture_canvas.app import GestureCanvasApp
from gesture_canvas.state import AIState

RED = (0, 0, 200)
GREEN = (0, 200, 0)


class FakeDetector:
    """Stands in for HandDetector, returning a scripted pose and landmarks."""

    LANDMARK_COUNT = 21

    def __init__(self, *args, **kwargs) -> None:
        self.fingers = [0, 0, 0, 0, 0]
        self.landmarks: list[list[int]] = []

    def find_hands(self, frame, draw=False):
        return frame

    def find_position(self, frame, hand_no=0, draw=False):
        return self.landmarks, None

    def fingers_up(self) -> list[int]:
        return self.fingers

    def close(self) -> None:
        pass


@pytest.fixture
def app(monkeypatch):
    # Never construct the real detector: it needs MediaPipe and a working GPU/
    # CPU graph, neither of which these tests exercise.
    monkeypatch.setattr("gesture_canvas.app.HandDetector", FakeDetector)
    instance = GestureCanvasApp(header_asset=None)
    instance.state.hand_detected = True
    return instance


def landmarks_at(index_xy, thumb_xy=(0, 0), middle_xy=None):
    """Build a 21-landmark list with the tips we care about placed."""
    middle_xy = middle_xy or (index_xy[0] + 30, index_xy[1])
    points = [[i, 0, 0] for i in range(21)]
    points[4] = [4, thumb_xy[0], thumb_xy[1]]
    points[8] = [8, index_xy[0], index_xy[1]]
    points[12] = [12, middle_xy[0], middle_xy[1]]
    return points


def blank_frame():
    return np.zeros((config.CAM_HEIGHT, config.CAM_WIDTH, 3), np.uint8)


def drag(app, points, fingers=(0, 1, 0, 0, 0)):
    """Move the index finger through a path with the given pose held."""
    app.detector.fingers = list(fingers)
    for point in points:
        app._dispatch(blank_frame(), landmarks_at(point))


def lift(app):
    """Return to a neutral pose, ending the current stroke.

    Four fingers up is the eraser, so a pose that maps to no gesture is used.
    """
    app.detector.fingers = [0, 0, 0, 0, 1]
    app._dispatch(blank_frame(), landmarks_at((900, 650)))


# ── Drawing ──────────────────────────────────────────────────────────────────
def test_index_finger_draws_on_the_canvas(app):
    drag(app, [(400, 300), (500, 300), (600, 300)])
    assert app.layers.base.any()


def test_drawing_uses_the_selected_colour(app):
    app.state.draw_color = GREEN

    drag(app, [(400, 300), (500, 300), (600, 300)])

    inked = app.layers.base[app.layers.base.any(axis=2)]
    assert inked[:, 1].max() > inked[:, 2].max()


def test_drawing_is_blocked_inside_the_header(app):
    drag(app, [(400, 40), (500, 40), (600, 40)])
    assert not app.layers.base.any()


def test_eraser_removes_existing_ink(app):
    drag(app, [(400, 300), (600, 300)])
    assert app.layers.base.any()

    app.state.active_tool = "eraser"
    drag(app, [(400, 300), (600, 300)])

    assert not app.layers.base[295:305, 450:550].any()


def test_a_stroke_pushes_exactly_one_undo_entry(app):
    """A continuous stroke is one undo step, not one per frame."""
    drag(app, [(400, 300), (450, 300), (500, 300), (550, 300)])
    assert len(app.undo) == 1


def test_separate_strokes_are_separate_undo_steps(app):
    drag(app, [(400, 300), (500, 300)])
    lift(app)
    drag(app, [(400, 400), (500, 400)])

    assert len(app.undo) == 2


def test_undo_reverts_a_stroke(app):
    drag(app, [(400, 300), (500, 300), (600, 300)])

    app._undo()

    assert not app.layers.base.any()


def test_no_hand_ends_the_stroke(app):
    drag(app, [(400, 300), (500, 300)])
    app.state.hand_detected = False

    app._dispatch(blank_frame(), [])

    assert not app.state.stroke_started
    assert len(app.state.current_stroke) > 2


# ── Four-finger gesture eraser ───────────────────────────────────────────────
FOUR_FINGERS = (0, 1, 1, 1, 1)


def _erase_at(app, point, frames=1):
    """Hold the four-finger pose with the *middle* fingertip on ``point``."""
    app.detector.fingers = list(FOUR_FINGERS)
    for _ in range(frames):
        app._dispatch(blank_frame(), landmarks_at((point[0] - 30, point[1]), middle_xy=point))


def test_four_fingers_erase_existing_ink(app):
    drag(app, [(400, 300), (600, 300)])
    assert app.layers.base.any()

    _erase_at(app, (500, 300))

    assert not app.layers.base[295:305, 495:505].any()


def test_gesture_eraser_does_not_change_the_selected_tool(app):
    """Unlike the toolbar eraser, the pose is transient — brush stays selected."""
    _erase_at(app, (500, 300))
    assert app.state.active_tool == "brush"


def test_gesture_erase_is_undoable(app):
    drag(app, [(400, 300), (600, 300)])
    lift(app)
    before = app.layers.base.copy()

    _erase_at(app, (500, 300))
    app._undo()

    assert np.array_equal(app.layers.base, before)


def test_continuous_erase_pushes_one_undo_entry(app):
    """Holding the pose is one undo step, not one per frame."""
    drag(app, [(400, 300), (600, 300)])
    lift(app)
    assert len(app.undo) == 1

    _erase_at(app, (500, 300), frames=6)

    assert len(app.undo) == 2


def test_erasing_is_blocked_inside_the_header(app):
    import cv2
    cv2.circle(app.layers.base, (500, 40), 30, RED, cv2.FILLED)
    before = app.layers.base.copy()

    _erase_at(app, (500, 40))

    assert np.array_equal(app.layers.base, before)


def test_erase_ends_an_in_progress_stroke(app):
    drag(app, [(400, 300), (500, 300)])
    _erase_at(app, (900, 650))

    assert not app.state.stroke_started


# ── Toolbar selection ────────────────────────────────────────────────────────
def _hold_select(app, x, y, frames=1):
    app.detector.fingers = [0, 1, 1, 0, 0]
    for _ in range(frames):
        app._dispatch(blank_frame(), landmarks_at((x, y)))


def test_two_finger_tap_switches_tool(app):
    start, end = config.TOOL_ZONES["eraser"]

    _hold_select(app, (start + end) // 2, 40)

    assert app.state.active_tool == "eraser"


def test_two_finger_tap_selects_a_colour(app):
    (start, end), color = config.COLOR_ZONES[2]

    _hold_select(app, (start + end) // 2, 40)

    assert app.state.draw_color == color


def test_lingering_on_a_button_registers_one_click(app):
    """Holding the pose over Undo must not undo repeatedly."""
    drag(app, [(400, 300), (500, 300)])
    lift(app)
    drag(app, [(400, 400), (500, 400)])
    assert len(app.undo) == 2

    start, end = config.TOOL_ZONES["undo"]
    _hold_select(app, (start + end) // 2, 40, frames=config.CLICK_COOLDOWN - 1)

    assert len(app.undo) == 1, "cooldown failed; multiple undos fired"


def test_clear_empties_the_canvas(app):
    drag(app, [(400, 300), (500, 300)])
    start, end = config.TOOL_ZONES["clear"]

    _hold_select(app, (start + end) // 2, 40)

    assert not app.layers.base.any()


def test_selection_in_the_canvas_area_does_not_draw(app):
    drag(app, [(400, 300), (500, 300)], fingers=(0, 1, 1, 0, 0))
    assert not app.layers.base.any()


# ── Resize ───────────────────────────────────────────────────────────────────
def _engage_pinch(app):
    """Spend one frame with the fingertips closed, which is what arms a resize.

    A pinch must start closed so that a thumb resting open never hijacks the
    drawing pose; once armed it stays armed while the pose is held.
    """
    app.detector.fingers = [1, 1, 0, 0, 0]
    app._dispatch(blank_frame(), landmarks_at((500, 400), thumb_xy=(500, 430)))


def _hold_pinch(app, thumb_xy, frames, index_xy=(500, 400)):
    for _ in range(frames):
        app._dispatch(blank_frame(), landmarks_at(index_xy, thumb_xy=thumb_xy))


def test_pinch_resizes_the_brush(app):
    app.state.brush_thickness = 15

    _engage_pinch(app)
    _hold_pinch(app, (500, 580), config.RESIZE_DEBOUNCE + 3)

    assert app.state.brush_thickness > 15


def test_pinch_resizes_the_eraser_when_it_is_active(app):
    app.state.active_tool = "eraser"

    _engage_pinch(app)
    _hold_pinch(app, (500, 560), config.RESIZE_DEBOUNCE + 3)

    assert config.MIN_ERASER <= app.state.eraser_thickness <= config.MAX_ERASER
    assert app.state.brush_thickness == config.DEFAULT_BRUSH


def test_resize_ignores_the_first_few_frames(app):
    """Fingers pass through the pinch pose in transit; that must not resize."""
    app.state.brush_thickness = 15

    _engage_pinch(app)  # counts as the first frame of the debounce
    _hold_pinch(app, (500, 580), config.RESIZE_DEBOUNCE - 2)

    assert app.state.brush_thickness == 15


def test_resize_is_clamped_to_the_configured_range(app):
    _engage_pinch(app)
    _hold_pinch(app, (1200, 700), config.RESIZE_DEBOUNCE + 3, index_xy=(100, 100))

    assert app.state.brush_thickness <= config.MAX_BRUSH


# ── The thumb must not steal the writing pose ────────────────────────────────
def test_open_thumb_draws_instead_of_resizing(app):
    """The old app ignored the thumb entirely; writing with it out must work."""
    app.detector.fingers = [1, 1, 0, 0, 0]
    for x in range(400, 700, 30):
        app._dispatch(blank_frame(), landmarks_at((x, 400), thumb_xy=(x - 150, 520)))

    assert app.layers.base.any(), "an open thumb blocked drawing"
    assert app.state.brush_thickness == config.DEFAULT_BRUSH, "it resized instead"


def test_spreading_an_engaged_pinch_keeps_resizing(app):
    """Growing the brush means spreading the fingers; that must not start a stroke."""
    _engage_pinch(app)
    _hold_pinch(app, (500, 620), config.RESIZE_DEBOUNCE + 3)

    assert app.state.brush_thickness > config.MIN_BRUSH
    assert not app.layers.base.any(), "resize leaked ink onto the canvas"


def test_releasing_the_pinch_disengages_it(app):
    """After the pose ends, an open thumb must draw again rather than resize."""
    _engage_pinch(app)
    _hold_pinch(app, (500, 580), 3)

    app.detector.fingers = [0, 1, 0, 0, 0]
    app._dispatch(blank_frame(), landmarks_at((300, 300)))
    assert not app._pinch_engaged

    app.detector.fingers = [1, 1, 0, 0, 0]
    for x in range(400, 700, 30):
        app._dispatch(blank_frame(), landmarks_at((x, 400), thumb_xy=(x - 150, 520)))

    assert app.layers.base.any()


# ── AI snap ──────────────────────────────────────────────────────────────────
def _circle_points(cx=640, cy=400, radius=120, step=6):
    import math
    return [
        (int(cx + radius * math.cos(math.radians(d))), int(cy + radius * math.sin(math.radians(d))))
        for d in range(0, 352, step)
    ]


def test_ai_arms_then_waits_after_a_stroke(app):
    app._toggle_ai()
    assert app.state.ai_state is AIState.ARMED

    drag(app, _circle_points())
    app.detector.fingers = [0, 0, 0, 0, 0]
    app._dispatch(blank_frame(), landmarks_at((500, 400)))

    assert app.state.ai_state is AIState.WAITING


def test_fist_must_be_held_to_confirm(app):
    app._toggle_ai()
    drag(app, _circle_points())

    app.detector.fingers = [0, 0, 0, 0, 0]
    for _ in range(config.FIST_CONFIRM - 1):
        app._dispatch(blank_frame(), landmarks_at((500, 400)))

    assert app.state.ai_state is AIState.WAITING, "snapped before the hold completed"


def test_held_fist_snaps_the_shape(app):
    app._toggle_ai()
    drag(app, _circle_points())

    app.detector.fingers = [0, 0, 0, 0, 0]
    for _ in range(config.FIST_CONFIRM + 1):
        app._dispatch(blank_frame(), landmarks_at((500, 400)))

    assert app.state.ai_state is AIState.ANIMATING
    assert "Circle" in app.state.ai_result


def test_snapping_a_scribble_keeps_the_drawing(app):
    """An unrecognisable stroke must be left exactly as drawn."""
    import random
    rng = random.Random(3)
    scribble = [(640 + rng.randint(-150, 150), 400 + rng.randint(-150, 150)) for _ in range(80)]

    app._toggle_ai()
    drag(app, scribble)
    before = app.layers.base.copy()

    app.detector.fingers = [0, 0, 0, 0, 0]
    for _ in range(config.FIST_CONFIRM + 1):
        app._dispatch(blank_frame(), landmarks_at((500, 400)))

    assert app.state.ai_state is AIState.IDLE
    assert np.array_equal(app.layers.base, before)


def test_toggling_ai_off_cancels_the_preview(app):
    app._toggle_ai()
    drag(app, _circle_points())
    app.detector.fingers = [0, 0, 0, 0, 0]
    for _ in range(config.FIST_CONFIRM + 1):
        app._dispatch(blank_frame(), landmarks_at((500, 400)))

    app._toggle_ai()

    assert app.state.ai_state is AIState.IDLE
    assert app.layers.preview is None


# ── Fill ─────────────────────────────────────────────────────────────────────
def test_three_finger_gesture_fills_a_closed_shape(app):
    import cv2
    cv2.circle(app.layers.base, (640, 400), 150, RED, 8, cv2.LINE_AA)
    app.state.draw_color = GREEN

    app.detector.fingers = [0, 1, 1, 1, 0]
    app._dispatch(blank_frame(), landmarks_at((640, 400)))

    assert tuple(app.layers.base[400, 640]) == GREEN


def test_fill_on_an_open_shape_reports_and_changes_nothing(app):
    import cv2
    cv2.ellipse(app.layers.base, (640, 400), (150, 150), 0, 30, 330, RED, 8, cv2.LINE_AA)
    before = app.layers.base.copy()
    app.state.draw_color = GREEN

    app.detector.fingers = [0, 1, 1, 1, 0]
    app._dispatch(blank_frame(), landmarks_at((640, 400)))

    assert np.array_equal(app.layers.base, before)
    assert "not closed" in app.state.ai_result


def test_successful_fill_is_undoable(app):
    import cv2
    cv2.circle(app.layers.base, (640, 400), 150, RED, 8, cv2.LINE_AA)
    before = app.layers.base.copy()
    app.state.draw_color = GREEN

    app.detector.fingers = [0, 1, 1, 1, 0]
    app._dispatch(blank_frame(), landmarks_at((640, 400)))
    app._undo()

    assert np.array_equal(app.layers.base, before)


# ── Full frame pipeline ──────────────────────────────────────────────────────
def _camera_frame():
    """A non-black frame, so canvas keying is actually exercised."""
    return np.full((480, 640, 3), 90, np.uint8)


def test_process_frame_returns_a_display_sized_image(app):
    """Frames are resized, so a camera that ignores the request still lines up."""
    result = app.process_frame(_camera_frame())
    assert result.shape == (config.CAM_HEIGHT, config.CAM_WIDTH, 3)


def test_process_frame_draws_the_toolbar(app):
    result = app.process_frame(_camera_frame())
    assert result[0 : config.HEADER_HEIGHT].any()


def test_process_frame_without_a_hand_is_stable(app):
    app.detector.landmarks = []
    for _ in range(5):
        app.process_frame(_camera_frame())

    assert not app.state.hand_detected
    assert not app.layers.base.any()


def test_process_frame_draws_when_a_hand_is_present(app):
    app.detector.fingers = [0, 1, 0, 0, 0]
    for x in range(400, 700, 40):
        app.detector.landmarks = landmarks_at((x, 400))
        app.process_frame(_camera_frame())

    assert app.state.hand_detected
    assert app.layers.base.any()


def test_canvas_ink_appears_in_the_display(app):
    import cv2
    cv2.circle(app.layers.base, (640, 400), 60, GREEN, cv2.FILLED)
    app.detector.landmarks = []

    result = app.process_frame(_camera_frame())

    assert tuple(result[400, 640]) == GREEN


def test_fps_readout_becomes_positive(app):
    app.detector.landmarks = []
    for _ in range(3):
        app.process_frame(_camera_frame())
    assert app.state.fps > 0


# ── Keyboard ─────────────────────────────────────────────────────────────────
def test_q_quits(app):
    assert app._handle_key(ord("q")) is False


def test_escape_quits(app):
    assert app._handle_key(27) is False


def test_z_undoes(app):
    drag(app, [(400, 300), (500, 300)])
    app._handle_key(ord("z"))
    assert not app.layers.base.any()


def test_s_writes_the_canvas(app, tmp_path):
    app.save_path = tmp_path / "out.png"
    drag(app, [(400, 300), (500, 300)])

    app._handle_key(ord("s"))

    assert app.save_path.is_file()
    assert app.state.save_msg_timer > 0


def test_unknown_key_is_ignored(app):
    assert app._handle_key(ord("j")) is True
