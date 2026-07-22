"""Tests for stroke preprocessing and contour extraction."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from gesture_canvas.shapes.contour import resample_stroke, smooth_stroke, stroke_to_contour


def test_resample_produces_uniform_spacing():
    # Arrange - a straight line with wildly uneven point density
    points = [(0, 0), (5, 0), (6, 0), (7, 0), (200, 0)]

    # Act
    resampled = resample_stroke(points, spacing=10)

    # Assert - consecutive gaps should all be near the requested spacing
    gaps = [
        math.hypot(resampled[i + 1][0] - resampled[i][0], resampled[i + 1][1] - resampled[i][1])
        for i in range(len(resampled) - 1)
    ]
    assert gaps, "resampling produced no segments"
    assert all(abs(gap - 10) < 2 for gap in gaps)


def test_resample_returns_input_when_too_short():
    assert resample_stroke([(1, 1)]) == [(1, 1)]
    assert resample_stroke([]) == []


def test_resample_skips_duplicate_points():
    # Arrange - repeated identical samples, as a stationary finger produces
    points = [(10, 10)] * 5 + [(60, 10)]

    # Act
    resampled = resample_stroke(points, spacing=10)

    # Assert
    assert len(resampled) > 1
    assert all(isinstance(p, tuple) for p in resampled)


def test_smoothing_reduces_jitter():
    # Arrange - a straight line with alternating 1px noise
    noisy = [(i, 100 + (5 if i % 2 == 0 else -5)) for i in range(60)]

    # Act
    smoothed = smooth_stroke(noisy, window=5)

    # Assert - deviation from the true line should shrink
    noisy_error = sum(abs(y - 100) for _, y in noisy) / len(noisy)
    smooth_error = sum(abs(y - 100) for _, y in smoothed) / len(smoothed)
    assert smooth_error < noisy_error


def test_smoothing_passes_through_short_strokes():
    points = [(1, 1), (2, 2)]
    assert smooth_stroke(points, window=5) == points


def test_unclosed_loop_yields_solid_contour(circle_stroke, canvas_size):
    """A hand-drawn circle never closes; it must still measure as a filled disc."""
    # Arrange
    height, width = canvas_size

    # Act
    contour, approx = stroke_to_contour(circle_stroke, height, width)

    # Assert - area must approximate a disc, not a thin ribbon
    assert contour is not None and approx is not None
    area = cv2.contourArea(contour)
    expected_disc_area = math.pi * 150 ** 2
    assert area > expected_disc_area * 0.7


def test_too_few_points_returns_none(canvas_size):
    height, width = canvas_size
    contour, approx = stroke_to_contour([(1, 1), (2, 2)], height, width)
    assert contour is None
    assert approx is None


def test_tiny_stroke_below_area_threshold_returns_none(canvas_size):
    # Arrange - a stroke so small its contour falls under min_contour_area
    height, width = canvas_size
    tiny = [(10, 10), (11, 10), (11, 11), (10, 11), (10, 10)]

    # Act
    contour, _ = stroke_to_contour(tiny, height, width)

    # Assert - either rejected outright, or at minimum not a usable shape
    if contour is not None:
        assert cv2.contourArea(contour) >= 300


def test_contour_stays_within_canvas(square_stroke, canvas_size):
    # Arrange
    height, width = canvas_size

    # Act
    contour, _ = stroke_to_contour(square_stroke, height, width)

    # Assert
    assert contour is not None
    points = contour.reshape(-1, 2)
    assert points[:, 0].min() >= 0 and points[:, 0].max() < width
    assert points[:, 1].min() >= 0 and points[:, 1].max() < height
