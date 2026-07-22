"""Tests for shape classification and clean rendering."""

from __future__ import annotations

import numpy as np
import pytest

from gesture_canvas.shapes.classify import FREEFORM, classify_shape
from gesture_canvas.shapes.contour import stroke_to_contour
from gesture_canvas.shapes.render import draw_clean_shape


def _classify(stroke, canvas_size):
    height, width = canvas_size
    contour, approx = stroke_to_contour(stroke, height, width)
    assert contour is not None, "fixture failed to produce a contour"
    name, confidence = classify_shape(contour, approx)
    return name, confidence, contour, approx


# ── Rounded shapes ───────────────────────────────────────────────────────────
def test_circle_is_recognised(circle_stroke, canvas_size):
    name, confidence, _, _ = _classify(circle_stroke, canvas_size)
    assert name == "Circle"
    assert confidence > 0.62


def test_ellipse_is_distinguished_from_circle(ellipse_stroke, canvas_size):
    name, confidence, _, _ = _classify(ellipse_stroke, canvas_size)
    assert name == "Ellipse"
    assert confidence > 0.62


# ── Straight-edged shapes ────────────────────────────────────────────────────
def test_triangle_is_recognised(triangle_stroke, canvas_size):
    name, confidence, _, _ = _classify(triangle_stroke, canvas_size)
    assert name == "Triangle"
    assert confidence > 0.62


def test_square_is_recognised(square_stroke, canvas_size):
    name, _, _, _ = _classify(square_stroke, canvas_size)
    assert name in {"Square", "Rectangle"}


def test_square_beats_rectangle_on_equal_sides(square_stroke, canvas_size):
    """A square must not lose to Rectangle, which shares its base score."""
    from gesture_canvas.shapes.classify import score_all
    from gesture_canvas.shapes.features import ShapeFeatures

    _, _, contour, approx = _classify(square_stroke, canvas_size)
    scores = score_all(ShapeFeatures.extract(contour, approx))

    assert scores["Square"] >= scores["Rectangle"]


def test_wide_rectangle_is_not_called_square(rectangle_stroke, canvas_size):
    from gesture_canvas.shapes.classify import score_all
    from gesture_canvas.shapes.features import ShapeFeatures

    _, _, contour, approx = _classify(rectangle_stroke, canvas_size)
    scores = score_all(ShapeFeatures.extract(contour, approx))

    assert scores["Rectangle"] > scores["Square"]


def test_pentagon_is_recognised(pentagon_stroke, canvas_size):
    name, confidence, _, _ = _classify(pentagon_stroke, canvas_size)
    assert name == "Pentagon"
    assert confidence > 0.62


def test_hexagon_is_recognised(hexagon_stroke, canvas_size):
    name, confidence, _, _ = _classify(hexagon_stroke, canvas_size)
    assert name == "Hexagon"
    assert confidence > 0.62


def test_star_is_recognised(star_stroke, canvas_size):
    name, confidence, _, _ = _classify(star_stroke, canvas_size)
    assert name == "Star"
    assert confidence > 0.62


# ── Lines ────────────────────────────────────────────────────────────────────
def test_horizontal_line_is_recognised(line_stroke, canvas_size):
    name, _, _, _ = _classify(line_stroke, canvas_size)
    assert name == "Line"


def test_diagonal_line_is_recognised(diagonal_line_stroke, canvas_size):
    """Guards the oriented-bounding-box fix: a diagonal is a Line, not a Square."""
    name, _, _, _ = _classify(diagonal_line_stroke, canvas_size)
    assert name == "Line"


# ── Rejection ────────────────────────────────────────────────────────────────
def test_scribble_is_left_alone(scribble_stroke, canvas_size):
    """Random noise must not be forced into a shape the user never drew."""
    name, confidence, _, _ = _classify(scribble_stroke, canvas_size)
    assert name == FREEFORM
    assert confidence == 0.0


def test_confidence_is_zero_only_for_freeform(circle_stroke, canvas_size):
    name, confidence, _, _ = _classify(circle_stroke, canvas_size)
    assert (confidence == 0.0) == (name == FREEFORM)


# ── Rendering ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "shape_name",
    ["Circle", "Ellipse", "Square", "Rectangle", "Triangle",
     "Pentagon", "Hexagon", "Star", "Heart", "Line", "Freeform"],
)
def test_every_renderer_marks_the_canvas(shape_name, circle_stroke, canvas_size):
    """Regression guard for np.int0, removed in NumPy 2.x — Rectangle used to crash."""
    # Arrange
    height, width = canvas_size
    contour, approx = stroke_to_contour(circle_stroke, height, width)
    canvas = np.zeros((height, width, 3), np.uint8)

    # Act
    draw_clean_shape(canvas, shape_name, contour, approx, (0, 0, 255), 8)

    # Assert
    assert canvas.any(), f"{shape_name} renderer drew nothing"


def test_render_of_unknown_shape_falls_back_to_hull(circle_stroke, canvas_size):
    height, width = canvas_size
    contour, approx = stroke_to_contour(circle_stroke, height, width)
    canvas = np.zeros((height, width, 3), np.uint8)

    draw_clean_shape(canvas, "NotAShape", contour, approx, (0, 255, 0), 6)

    assert canvas.any()


def test_rendered_shape_uses_requested_colour(circle_stroke, canvas_size):
    height, width = canvas_size
    contour, approx = stroke_to_contour(circle_stroke, height, width)
    canvas = np.zeros((height, width, 3), np.uint8)

    draw_clean_shape(canvas, "Circle", contour, approx, (0, 0, 255), 8)

    inked = canvas[canvas.any(axis=2)]
    assert (inked[:, 2] > 0).all(), "red channel missing from a red shape"
