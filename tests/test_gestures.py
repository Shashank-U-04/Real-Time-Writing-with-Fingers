"""Tests for gesture classification.

The finger vector is ``[thumb, index, middle, ring, pinky]``.
"""

from __future__ import annotations

import pytest

from gesture_canvas import config
from gesture_canvas.gestures import Gesture, classify


@pytest.mark.parametrize(
    "fingers,expected",
    [
        ([0, 1, 0, 0, 0], Gesture.DRAW),
        ([1, 1, 0, 0, 0], Gesture.RESIZE),
        ([0, 1, 1, 0, 0], Gesture.SELECT),
        ([0, 1, 1, 1, 0], Gesture.FILL),
        ([0, 0, 0, 0, 0], Gesture.FIST),
        ([0, 1, 1, 1, 1], Gesture.ERASE),
        ([0, 0, 0, 0, 1], Gesture.NONE),
    ],
)
def test_finger_poses_map_to_gestures(fingers, expected):
    assert classify(fingers) == expected


def test_four_fingers_erase_regardless_of_the_thumb():
    """The original finger vector had no thumb, so a flat hand always erased."""
    assert classify([0, 1, 1, 1, 1]) == Gesture.ERASE
    assert classify([1, 1, 1, 1, 1]) == Gesture.ERASE


def test_erase_wins_over_fill():
    """Fill needs the pinky down; raising it must switch to the eraser."""
    assert classify([0, 1, 1, 1, 0]) == Gesture.FILL
    assert classify([0, 1, 1, 1, 1]) == Gesture.ERASE


def test_pinch_wins_over_selection():
    """Thumb+index shares fingers with selection; the pinch must be preferred."""
    assert classify([1, 1, 0, 0, 0]) == Gesture.RESIZE


# ── Thumb must not steal the writing pose ────────────────────────────────────
def test_closed_pinch_resizes():
    assert classify([1, 1, 0, 0, 0], pinch_distance=25.0) == Gesture.RESIZE


def test_open_thumb_still_draws():
    """The original had no thumb in its vector, so a raised thumb never
    interrupted writing. A thumb held well clear of the index must still draw."""
    assert classify([1, 1, 0, 0, 0], pinch_distance=180.0) == Gesture.DRAW


def test_pinch_engagement_threshold_is_respected():
    below = config.PINCH_ENGAGE_DIST - 1
    above = config.PINCH_ENGAGE_DIST + 1

    assert classify([1, 1, 0, 0, 0], pinch_distance=below) == Gesture.RESIZE
    assert classify([1, 1, 0, 0, 0], pinch_distance=above) == Gesture.DRAW


def test_unknown_distance_keeps_the_pose_only_reading():
    assert classify([1, 1, 0, 0, 0], pinch_distance=None) == Gesture.RESIZE


def test_distance_does_not_affect_other_gestures():
    """Only the thumb+index collision is distance-sensitive."""
    for distance in (10.0, 300.0, None):
        assert classify([0, 1, 0, 0, 0], distance) == Gesture.DRAW
        assert classify([0, 1, 1, 0, 0], distance) == Gesture.SELECT
        assert classify([0, 1, 1, 1, 0], distance) == Gesture.FILL
        assert classify([0, 1, 1, 1, 1], distance) == Gesture.ERASE
        assert classify([0, 0, 0, 0, 0], distance) == Gesture.FIST


def test_fill_wins_over_selection():
    """Index+middle+ring also satisfies the selection pose; fill is more specific."""
    assert classify([0, 1, 1, 1, 0]) == Gesture.FILL


def test_drawing_ignores_the_thumb():
    """A thumb drifting up mid-stroke must not break drawing."""
    assert classify([0, 1, 0, 1, 0]) == Gesture.DRAW


def test_thumb_alone_is_not_a_gesture():
    assert classify([1, 0, 0, 0, 0]) == Gesture.NONE


def test_malformed_vector_is_rejected():
    assert classify([1, 0, 0]) == Gesture.NONE
    assert classify([]) == Gesture.NONE
