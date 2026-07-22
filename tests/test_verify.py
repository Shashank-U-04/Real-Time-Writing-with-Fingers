"""Tests for shape fit verification.

Verification is what stops the classifier replacing a user's stroke with a shape
that merely scored well without resembling what they drew.
"""

from __future__ import annotations

import numpy as np
import pytest

from gesture_canvas.shapes.contour import stroke_to_contour
from gesture_canvas.shapes.verify import fit_iou


def _contour_of(stroke, canvas_size):
    height, width = canvas_size
    contour, approx = stroke_to_contour(stroke, height, width)
    assert contour is not None
    return contour, approx


def test_correct_shape_scores_high_overlap(circle_stroke, canvas_size):
    contour, approx = _contour_of(circle_stroke, canvas_size)
    assert fit_iou("Circle", contour, approx) > 0.9


def test_wrong_shape_scores_low_overlap(circle_stroke, canvas_size):
    """A triangle laid over a circle covers far too little of it."""
    contour, approx = _contour_of(circle_stroke, canvas_size)
    assert fit_iou("Triangle", contour, approx) < 0.8


def test_scribble_matches_nothing_well(scribble_stroke, canvas_size):
    contour, approx = _contour_of(scribble_stroke, canvas_size)
    for candidate in ("Circle", "Ellipse", "Square", "Triangle", "Heart", "Star"):
        assert fit_iou(candidate, contour, approx) < 0.72, f"{candidate} matched a scribble"


@pytest.mark.parametrize(
    "fixture_name,shape",
    [
        ("circle_stroke", "Circle"),
        ("ellipse_stroke", "Ellipse"),
        ("square_stroke", "Square"),
        ("rectangle_stroke", "Rectangle"),
        ("triangle_stroke", "Triangle"),
        ("pentagon_stroke", "Pentagon"),
        ("hexagon_stroke", "Hexagon"),
        ("star_stroke", "Star"),
    ],
)
def test_renderers_fit_their_own_shape(fixture_name, shape, request, canvas_size):
    """Each renderer must reproduce the shape it claims well enough to verify."""
    stroke = request.getfixturevalue(fixture_name)
    contour, approx = _contour_of(stroke, canvas_size)

    overlap = fit_iou(shape, contour, approx)

    assert overlap >= 0.8, f"{shape} renderer only reached IoU {overlap:.2f}"


def test_overlap_is_bounded(circle_stroke, canvas_size):
    contour, approx = _contour_of(circle_stroke, canvas_size)
    assert 0.0 <= fit_iou("Circle", contour, approx) <= 1.0


@pytest.mark.parametrize(
    "junk",
    [
        np.array([[[0, 0]]], dtype=np.int32),
        np.array([[[0, 0]], [[0, 0]]], dtype=np.int32),
        np.array([[[5, 5]], [[5, 6]], [[5, 7]]], dtype=np.int32),
    ],
)
def test_degenerate_contours_return_a_bounded_score(junk):
    """Junk input must yield a usable number, never an exception."""
    result = fit_iou("Circle", junk, junk)
    assert 0.0 <= result <= 1.0


def test_unknown_shape_name_does_not_raise(circle_stroke, canvas_size):
    """An unrecognised name renders as a hull rather than failing verification."""
    contour, approx = _contour_of(circle_stroke, canvas_size)
    assert 0.0 <= fit_iou("NotAShape", contour, approx) <= 1.0
