"""Smoke tests for the on-screen overlays.

These assert that each overlay draws something, within bounds, without raising.
A coordinate mistake here would crash the render loop on a live frame, so the
cheap check is worth having even though the visual result is not asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from gesture_canvas import config, overlay
from gesture_canvas.state import AIState, AppState


@pytest.fixture
def frame():
    return np.zeros((config.CAM_HEIGHT, config.CAM_WIDTH, 3), np.uint8)


@pytest.fixture
def state():
    return AppState()


@pytest.mark.parametrize("ai_state", list(AIState))
def test_ai_hint_renders_for_every_state(frame, state, ai_state):
    state.ai_state = ai_state
    state.fist_count = 4

    overlay.draw_ai_hint(frame, state)

    # IDLE has no hint text, so it is the one state that draws nothing.
    assert frame.any() == (ai_state is not AIState.IDLE)


def _progress_bar_width(state) -> int:
    """Width of the filled portion of the fist-hold bar, in pixels."""
    frame = np.zeros((config.CAM_HEIGHT, config.CAM_WIDTH, 3), np.uint8)
    overlay.draw_ai_hint(frame, state)

    accent = overlay.AI_COLORS[AIState.WAITING]
    row = frame[config.HEADER_HEIGHT + 10 + 36]
    return int(np.count_nonzero(np.all(row == accent, axis=1)))


def test_fist_progress_bar_grows_with_the_hold(state):
    state.ai_state = AIState.WAITING

    state.fist_count = 1
    early = _progress_bar_width(state)

    state.fist_count = config.FIST_CONFIRM
    late = _progress_bar_width(state)

    assert 0 < early < late


def test_fist_progress_bar_is_absent_before_the_hold_starts(state):
    state.ai_state = AIState.WAITING
    state.fist_count = 0

    assert _progress_bar_width(state) == 0


def test_settings_panel_renders(frame, state):
    state.settings_open = True
    overlay.draw_settings_panel(frame, state)
    assert frame.any()


def test_settings_panel_stays_inside_the_frame(frame, state):
    """Guards against the panel being clipped or overflowing the window."""
    overlay.draw_settings_panel(frame, state)

    inked = np.argwhere(frame.any(axis=2))
    assert inked[:, 0].max() < config.CAM_HEIGHT
    assert inked[:, 1].max() < config.CAM_WIDTH


def test_status_banner_renders_both_halves(frame):
    overlay.draw_status_banner(frame, "Snapped: Circle | Confidence 91%")
    assert frame.any()


def test_status_banner_handles_a_message_without_a_separator(frame):
    overlay.draw_status_banner(frame, "Area filled")
    assert frame.any()


def test_save_confirmation_renders(frame):
    overlay.draw_save_confirmation(frame)
    assert frame.any()


def test_resize_indicator_renders(frame):
    overlay.draw_resize_indicator(frame, (400, 500), (400, 300), 30, (0, 0, 200), locked=False)
    assert frame.any()


def test_resize_indicator_shows_a_locked_state(frame):
    unlocked = np.zeros_like(frame)
    overlay.draw_resize_indicator(unlocked, (400, 500), (400, 300), 30, (0, 0, 200), locked=False)

    overlay.draw_resize_indicator(frame, (400, 500), (400, 300), 30, (0, 0, 200), locked=True)

    assert not np.array_equal(frame, unlocked)


# ── Canvas keying ────────────────────────────────────────────────────────────
def test_keying_shows_the_camera_where_nothing_is_drawn():
    camera = np.full((100, 100, 3), 90, np.uint8)
    canvas = np.zeros((100, 100, 3), np.uint8)

    result = overlay.key_canvas_over_feed(camera, canvas)

    assert tuple(result[50, 50]) == (90, 90, 90)


def test_keying_shows_ink_over_the_camera():
    camera = np.full((100, 100, 3), 90, np.uint8)
    canvas = np.zeros((100, 100, 3), np.uint8)
    canvas[40:60, 40:60] = (0, 0, 200)

    result = overlay.key_canvas_over_feed(camera, canvas)

    assert tuple(result[50, 50]) == (0, 0, 200)


def test_keying_does_not_alter_the_frame_size():
    camera = np.full((config.CAM_HEIGHT, config.CAM_WIDTH, 3), 60, np.uint8)
    canvas = np.zeros_like(camera)

    assert overlay.key_canvas_over_feed(camera, canvas).shape == camera.shape
