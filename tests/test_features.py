"""Tests for the geometric feature extractors."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gesture_canvas.shapes import features as feat
from gesture_canvas.shapes.contour import stroke_to_contour


def _contour_of(stroke, canvas_size):
    height, width = canvas_size
    contour, approx = stroke_to_contour(stroke, height, width)
    assert contour is not None, "fixture failed to produce a contour"
    return contour, approx


def test_circle_has_high_circularity(circle_stroke, canvas_size):
    contour, _ = _contour_of(circle_stroke, canvas_size)
    assert feat.circularity(contour) > 0.85


def test_triangle_has_low_circularity(triangle_stroke, canvas_size):
    contour, _ = _contour_of(triangle_stroke, canvas_size)
    assert feat.circularity(contour) < 0.75


def test_star_has_low_solidity(star_stroke, canvas_size):
    """A star's points leave big voids against its convex hull."""
    contour, _ = _contour_of(star_stroke, canvas_size)
    assert feat.solidity(contour) < 0.75


def test_circle_has_high_solidity(circle_stroke, canvas_size):
    contour, _ = _contour_of(circle_stroke, canvas_size)
    assert feat.solidity(contour) > 0.9


def test_circle_ellipse_ratio_near_one(circle_stroke, canvas_size):
    contour, _ = _contour_of(circle_stroke, canvas_size)
    assert feat.ellipse_ratio(contour) < 1.2


def test_ellipse_ratio_detects_elongation(ellipse_stroke, canvas_size):
    contour, _ = _contour_of(ellipse_stroke, canvas_size)
    assert feat.ellipse_ratio(contour) > 1.5


def test_diagonal_line_has_high_oriented_aspect(diagonal_line_stroke, canvas_size):
    """The bug this guards: an upright box reads a 45-degree line as square."""
    contour, _ = _contour_of(diagonal_line_stroke, canvas_size)

    oriented = feat.min_area_aspect(contour)

    import cv2
    _, _, w, h = cv2.boundingRect(contour)
    upright = max(w, h) / max(min(w, h), 1e-6)

    assert oriented > 4.0
    assert oriented > upright


def test_square_oriented_aspect_near_one(square_stroke, canvas_size):
    contour, _ = _contour_of(square_stroke, canvas_size)
    assert abs(feat.min_area_aspect(contour) - 1.0) < 0.25


def test_star_has_multiple_defects(star_stroke, canvas_size):
    contour, _ = _contour_of(star_stroke, canvas_size)
    assert feat.defect_count(contour) >= 4


def test_circle_has_no_significant_defects(circle_stroke, canvas_size):
    contour, _ = _contour_of(circle_stroke, canvas_size)
    assert feat.defect_count(contour) <= 1


def test_symmetric_shape_scores_high(circle_stroke, canvas_size):
    contour, _ = _contour_of(circle_stroke, canvas_size)
    assert feat.bilateral_symmetry(contour) > 0.8


def test_interior_angles_of_square_near_ninety(square_stroke, canvas_size):
    contour, approx = _contour_of(square_stroke, canvas_size)
    if len(approx) == 4:
        angles = feat.interior_angles(approx)
        assert all(abs(a - 90) < 15 for a in angles)


def test_interior_angles_empty_for_degenerate_polygon():
    assert feat.interior_angles(np.array([[[0, 0]], [[1, 1]]])) == []


def test_features_extract_populates_every_field(square_stroke, canvas_size):
    contour, approx = _contour_of(square_stroke, canvas_size)

    result = feat.ShapeFeatures.extract(contour, approx)

    assert result.corners > 0
    assert 0.0 <= result.circularity <= 1.5
    assert 0.0 <= result.solidity <= 1.01
    assert result.min_aspect >= 1.0
    assert isinstance(result.angles, tuple)


def test_feature_extractors_survive_degenerate_contour():
    """Degenerate input must degrade to a neutral value, not raise."""
    degenerate = np.array([[[5, 5]], [[5, 6]], [[5, 7]]], dtype=np.int32)

    assert feat.ellipse_ratio(degenerate) == 1.0
    assert feat.defect_count(degenerate) >= 0
    assert 0.0 <= feat.bilateral_symmetry(degenerate) <= 1.0
    assert 0.0 <= feat.top_notch_depth(degenerate) <= 1.0
    assert 0.0 <= feat.bottom_point_sharpness(degenerate) <= 1.0
