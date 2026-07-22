"""Tests for the leak-proof flood fill."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from gesture_canvas.fill import flood_fill

HEADER = 100
HEIGHT, WIDTH = 720, 1280
RED = (0, 0, 200)
GREEN = (0, 200, 0)


@pytest.fixture
def blank():
    return np.zeros((HEIGHT, WIDTH, 3), np.uint8)


@pytest.fixture
def closed_circle(blank):
    """A closed circle outline, well clear of every canvas edge."""
    cv2.circle(blank, (640, 400), 150, RED, 8, cv2.LINE_AA)
    return blank


@pytest.fixture
def open_circle(blank):
    """The same circle with a gap in its outline."""
    cv2.ellipse(blank, (640, 400), (150, 150), 0, 30, 330, RED, 8, cv2.LINE_AA)
    return blank


def test_fill_inside_closed_shape_succeeds(closed_circle):
    filled, result = flood_fill(closed_circle, 640, 400, GREEN)

    assert result.success
    assert tuple(filled[400, 640]) == GREEN


def test_fill_stays_inside_the_shape(closed_circle):
    # Act
    filled, result = flood_fill(closed_circle, 640, 400, GREEN)

    # Assert - a point well outside the circle must remain untouched
    assert result.success
    assert tuple(filled[400, 200]) == (0, 0, 0)
    assert tuple(filled[650, 640]) == (0, 0, 0)


def test_open_shape_is_rejected(open_circle):
    original = open_circle.copy()

    filled, result = flood_fill(open_circle, 640, 400, GREEN)

    assert not result.success
    assert "not closed" in result.message
    assert np.array_equal(filled, original), "canvas must be untouched on abort"


def test_fill_does_not_mutate_the_input(closed_circle):
    before = closed_circle.copy()

    flood_fill(closed_circle, 640, 400, GREEN)

    assert np.array_equal(closed_circle, before)


def test_existing_strokes_keep_their_colour(closed_circle):
    """The fill slides under the outline; it must not recolour it."""
    filled, result = flood_fill(closed_circle, 640, 400, GREEN)

    assert result.success
    # The outline's own pixels should still be dominated by red, not green.
    outline_pixel = filled[400, 640 - 150]
    assert outline_pixel[2] > outline_pixel[1]


def test_no_halo_between_fill_and_stroke(closed_circle):
    """Every pixel just inside the outline must be inked, not left background."""
    filled, _ = flood_fill(closed_circle, 640, 400, GREEN)

    # Sample a ring just inside the stroke.
    for angle in range(0, 360, 15):
        x = int(640 + 141 * np.cos(np.radians(angle)))
        y = int(400 + 141 * np.sin(np.radians(angle)))
        assert filled[y, x].any(), f"gap left at {angle} degrees"


def test_seed_above_header_is_ignored(closed_circle):
    _, result = flood_fill(closed_circle, 640, HEADER - 10, GREEN)
    assert not result.success


def test_seed_outside_canvas_is_ignored(closed_circle):
    for x, y in [(-5, 400), (WIDTH + 5, 400), (640, HEIGHT + 5)]:
        _, result = flood_fill(closed_circle, x, y, GREEN)
        assert not result.success


def test_seed_on_a_stroke_is_rejected(closed_circle):
    _, result = flood_fill(closed_circle, 640, 400 - 150, GREEN)
    assert not result.success


def test_refilling_the_same_colour_is_a_no_op(closed_circle):
    filled, _ = flood_fill(closed_circle, 640, 400, GREEN)

    again, result = flood_fill(filled, 640, 400, GREEN)

    assert not result.success
    assert np.array_equal(again, filled)


def test_shape_closed_by_the_header_can_be_filled(blank):
    """A shape drawn up against the toolbar is sealed by it, not leaked through."""
    # Arrange - a U shape open at the top, where the header forms the lid
    cv2.line(blank, (400, HEADER), (400, 500), RED, 8)
    cv2.line(blank, (400, 500), (800, 500), RED, 8)
    cv2.line(blank, (800, 500), (800, HEADER), RED, 8)

    # Act
    filled, result = flood_fill(blank, 600, 300, GREEN)

    # Assert
    assert result.success, result.message
    assert tuple(filled[300, 600]) == GREEN
    assert tuple(filled[300, 200]) == (0, 0, 0)


def test_fill_region_reaches_the_boundary_but_not_past_it(blank):
    # Arrange - an axis-aligned box makes the expected extent exact
    cv2.rectangle(blank, (300, 200), (700, 500), RED, 6)

    # Act
    filled, result = flood_fill(blank, 500, 350, GREEN)

    # Assert
    assert result.success
    assert tuple(filled[350, 500]) == GREEN
    assert tuple(filled[350, 250]) == (0, 0, 0), "leaked left of the box"
    assert tuple(filled[150, 500]) == (0, 0, 0), "leaked above the box"
