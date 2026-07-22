"""Tests for toolbar hit-testing, hover debounce and click cooldown."""

from __future__ import annotations

import numpy as np
import pytest

from gesture_canvas import config
from gesture_canvas.toolbar import HoverTracker, Toolbar, hit_test


def _midpoint(zone: tuple[int, int]) -> int:
    return (zone[0] + zone[1]) // 2


# ── Hit testing ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("tool", ["brush", "eraser", "undo", "clear", "save", "ai", "settings"])
def test_each_tool_zone_resolves_to_its_tool(tool):
    x = _midpoint(config.TOOL_ZONES[tool])

    found, color = hit_test(x, config.HEADER_HEIGHT // 2)

    assert found == tool
    assert color is None


def test_points_below_the_header_hit_nothing():
    x = _midpoint(config.TOOL_ZONES["brush"])
    assert hit_test(x, config.HEADER_HEIGHT) == (None, None)
    assert hit_test(x, config.HEADER_HEIGHT + 50) == (None, None)


def test_points_outside_the_frame_hit_nothing():
    assert hit_test(-1, 10) == (None, None)
    assert hit_test(config.CAM_WIDTH, 10) == (None, None)


def test_every_swatch_resolves_to_its_colour():
    for (start, end), expected in config.COLOR_ZONES:
        found, color = hit_test((start + end) // 2, 40)
        assert found == "colors"
        assert color == expected


def test_gap_between_swatches_selects_no_colour():
    """A near-miss must not silently change the colour."""
    (_, first_end), _ = config.COLOR_ZONES[0]
    (second_start, _), _ = config.COLOR_ZONES[1]
    if second_start - first_end < 2:
        pytest.skip("swatches are adjacent in this layout")

    found, color = hit_test((first_end + second_start) // 2, 40)

    assert found == "colors"
    assert color is None


def test_zone_boundaries_do_not_overlap():
    """Adjacent zones must not both claim the same pixel."""
    for tool, (start, end) in config.TOOL_ZONES.items():
        assert hit_test(start, 40)[0] == tool
        # The last pixel belongs to this zone; the next one does not.
        assert hit_test(end - 1, 40)[0] == tool


def test_zones_cover_the_full_width_without_gaps():
    spans = sorted(config.TOOL_ZONES.values())
    assert spans[0][0] == 0
    assert spans[-1][1] == config.CAM_WIDTH
    for (_, end), (next_start, _) in zip(spans, spans[1:]):
        assert end == next_start, "gap or overlap between tool zones"


# ── Hover debounce ───────────────────────────────────────────────────────────
def test_hover_requires_sustained_presence():
    tracker = HoverTracker(debounce=5)
    x = _midpoint(config.TOOL_ZONES["save"])

    for _ in range(4):
        tracker.update(x, 40)
    assert tracker.tool is None, "hover registered before the debounce elapsed"

    for _ in range(2):
        tracker.update(x, 40)
    assert tracker.tool == "save"


def test_hover_jitter_between_zones_never_registers():
    """Alternating across a boundary must not light either zone up."""
    tracker = HoverTracker(debounce=5)
    left = _midpoint(config.TOOL_ZONES["undo"])
    right = _midpoint(config.TOOL_ZONES["clear"])

    for i in range(20):
        tracker.update(left if i % 2 == 0 else right, 40)

    assert tracker.tool is None


def test_hover_clears_when_the_hand_leaves():
    tracker = HoverTracker(debounce=2)
    x = _midpoint(config.TOOL_ZONES["ai"])
    for _ in range(5):
        tracker.update(x, 40)
    assert tracker.tool == "ai"

    tracker.clear()

    assert tracker.tool is None


# ── Click cooldown ───────────────────────────────────────────────────────────
def test_click_returns_the_tool():
    bar = Toolbar()
    tool, _ = bar.click(_midpoint(config.TOOL_ZONES["eraser"]), 40)
    assert tool == "eraser"


def test_repeat_clicks_are_suppressed_during_cooldown():
    """A fingertip rests on a button for many frames; only the first counts."""
    bar = Toolbar()
    x = _midpoint(config.TOOL_ZONES["undo"])

    first, _ = bar.click(x, 40)
    repeats = [bar.click(x, 40)[0] for _ in range(config.CLICK_COOLDOWN - 1)]

    assert first == "undo"
    assert all(r is None for r in repeats)


def test_clicks_resume_after_the_cooldown_elapses():
    bar = Toolbar()
    x = _midpoint(config.TOOL_ZONES["undo"])
    bar.click(x, 40)

    for _ in range(config.CLICK_COOLDOWN):
        bar.tick()

    assert bar.click(x, 40)[0] == "undo"


def test_clicking_a_swatch_gap_is_not_a_click():
    bar = Toolbar()
    (_, first_end), _ = config.COLOR_ZONES[0]
    (second_start, _), _ = config.COLOR_ZONES[1]
    if second_start - first_end < 2:
        pytest.skip("swatches are adjacent in this layout")

    tool, color = bar.click((first_end + second_start) // 2, 40)

    assert tool is None and color is None
    assert bar.accepting_clicks, "a missed click must not burn the cooldown"


# ── Rendering ────────────────────────────────────────────────────────────────
def test_toolbar_renders_without_a_background_asset():
    """The app must run with no header image present."""
    bar = Toolbar(background_path=None)
    frame = np.zeros((config.CAM_HEIGHT, config.CAM_WIDTH, 3), np.uint8)

    bar.draw(frame, "brush", config.DEFAULT_COLOR)

    assert frame[0 : config.HEADER_HEIGHT].any(), "header area was left blank"
    assert not frame[config.HEADER_HEIGHT :].any(), "toolbar drew below the header"


def test_missing_background_file_is_not_an_error(tmp_path):
    bar = Toolbar(background_path=tmp_path / "does_not_exist.png")
    assert bar.background is None


def test_background_image_is_resized_to_the_header(tmp_path):
    import cv2

    path = tmp_path / "header.png"
    cv2.imwrite(str(path), np.full((50, 50, 3), 120, np.uint8))

    bar = Toolbar(background_path=path)

    assert bar.background is not None
    assert bar.background.shape == (config.HEADER_HEIGHT, config.CAM_WIDTH, 3)


def test_active_tool_is_visually_distinguished():
    frame_a = np.zeros((config.HEADER_HEIGHT, config.CAM_WIDTH, 3), np.uint8)
    frame_b = np.zeros((config.HEADER_HEIGHT, config.CAM_WIDTH, 3), np.uint8)

    Toolbar().draw(frame_a, "brush", config.DEFAULT_COLOR)
    Toolbar().draw(frame_b, "eraser", config.DEFAULT_COLOR)

    assert not np.array_equal(frame_a, frame_b)
