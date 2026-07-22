"""Draw the idealised version of a recognised shape.

Each renderer fits clean geometry to the user's contour rather than tracing it,
so the result is a true circle / rectangle / star positioned and sized where the
user drew.
"""

from __future__ import annotations

import logging
import math
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_RENDER_ERRORS = (cv2.error, ValueError, ZeroDivisionError, IndexError)

BGR = tuple[int, int, int]


def _as_int_points(points: list[list[int]]) -> np.ndarray:
    # np.int0 was removed in NumPy 2.0; np.intp is the supported spelling.
    return np.array(points, dtype=np.intp)


def _draw_polygon(canvas: np.ndarray, points: list[list[int]], color: BGR, thickness: int) -> None:
    """Draw a closed polygon, filling it when ``thickness`` is negative.

    Fill support exists so the same geometry can be rasterised as a solid mask
    for fit verification, keeping the verified shape identical to the drawn one.
    """
    array = np.array(points, np.int32)
    if thickness < 0:
        cv2.fillPoly(canvas, [array], color)
    else:
        cv2.polylines(canvas, [array.reshape((-1, 1, 2))], True, color, thickness, cv2.LINE_AA)


def _draw_heart(canvas: np.ndarray, contour: np.ndarray, color: BGR, thickness: int) -> None:
    """Plot the classic parametric heart, scaled into the contour's bounding box."""
    x, y, width, height = cv2.boundingRect(contour)
    center_x = x + width // 2

    points: list[list[int]] = []
    for degrees in range(0, 361, 4):
        t = math.radians(degrees)
        hx = 16 * (math.sin(t) ** 3)
        hy = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        # 36 and 28 are the parametric curve's own extents; dividing by them maps
        # the curve onto the bounding box.
        points.append([int(center_x + hx * width / 36), int(y + height // 2 + hy * height / 28)])

    _draw_polygon(canvas, points, color, thickness)


def _centroid(contour: np.ndarray) -> tuple[float, float]:
    """Area centroid, falling back to the bounding-box centre if degenerate.

    The centroid, not the bounding-box centre, is the right anchor for radially
    symmetric shapes: a five-pointed star's bounding box is offset from its
    true centre because the points are not vertically symmetric.
    """
    moments = cv2.moments(contour)
    if moments["m00"] > 1e-6:
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    x, y, width, height = cv2.boundingRect(contour)
    return x + width / 2, y + height / 2


def _symmetry_phase(contour: np.ndarray, center: tuple[float, float], folds: int) -> float:
    """Estimate the rotation of an ``folds``-fold symmetric shape, in radians.

    Takes the circular mean of ``folds * angle`` over the contour. Weighting by
    radius to the fourth power lets the vertices dominate the edge points, so
    the result locks onto the corners rather than the average outline.
    """
    points = contour.reshape(-1, 2).astype(float)
    offsets = points - np.array(center)
    radii = np.linalg.norm(offsets, axis=1)
    if radii.max() < 1e-6:
        return -math.pi / 2

    angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    weights = (radii / radii.max()) ** 4
    sin_sum = float(np.sum(weights * np.sin(folds * angles)))
    cos_sum = float(np.sum(weights * np.cos(folds * angles)))
    return math.atan2(sin_sum, cos_sum) / folds


def _draw_star(canvas: np.ndarray, contour: np.ndarray, color: BGR, thickness: int) -> None:
    """Five-pointed star fitted to the stroke's centre, size and rotation."""
    center = _centroid(contour)
    offsets = contour.reshape(-1, 2).astype(float) - np.array(center)
    radii = np.linalg.norm(offsets, axis=1)

    # Percentiles rather than min/max so a single stray contour point cannot
    # stretch or shrink the whole star.
    outer_radius = float(np.percentile(radii, 97))
    inner_radius = float(np.percentile(radii, 12))
    if inner_radius < outer_radius * 0.2 or inner_radius > outer_radius * 0.75:
        inner_radius = outer_radius / 2.5

    phase = _symmetry_phase(contour, center, folds=5)

    points: list[list[int]] = []
    for i in range(10):
        # Inner vertices sit halfway between consecutive outer ones.
        angle = phase + i * math.pi / 5
        radius = outer_radius if i % 2 == 0 else inner_radius
        points.append([
            int(center[0] + math.cos(angle) * radius),
            int(center[1] + math.sin(angle) * radius),
        ])

    _draw_polygon(canvas, points, color, thickness)


def _draw_line(canvas: np.ndarray, contour: np.ndarray, color: BGR, thickness: int) -> None:
    """Total-least-squares fit through the stroke, drawn to its full extent."""
    _, _, width, height = cv2.boundingRect(contour)
    # fitLine returns a column vector in OpenCV 4 and a flat array in 5; ravel
    # normalises both before unpacking.
    vx, vy, x0, y0 = cv2.fitLine(
        contour.reshape(-1, 2).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
    ).ravel().tolist()
    half_length = max(width, height) // 2 + 10
    start = (int(x0 - vx * half_length), int(y0 - vy * half_length))
    end = (int(x0 + vx * half_length), int(y0 + vy * half_length))
    cv2.line(canvas, start, end, color, thickness, cv2.LINE_AA)


def _draw_circle(canvas: np.ndarray, contour: np.ndarray, color: BGR, thickness: int) -> None:
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    cv2.circle(canvas, (int(cx), int(cy)), int(radius), color, thickness, cv2.LINE_AA)


def _draw_ellipse(canvas: np.ndarray, contour: np.ndarray, color: BGR, thickness: int) -> None:
    if len(contour) >= 5:
        cv2.ellipse(canvas, cv2.fitEllipse(contour), color, thickness, cv2.LINE_AA)
        return
    x, y, width, height = cv2.boundingRect(contour)
    cv2.ellipse(canvas, (x + width // 2, y + height // 2), (width // 2, height // 2),
                0, 0, 360, color, thickness, cv2.LINE_AA)


def _draw_square(canvas: np.ndarray, contour: np.ndarray, color: BGR, thickness: int) -> None:
    """Force equal sides while keeping the stroke's centre and orientation."""
    (cx, cy), (width, height), angle = cv2.minAreaRect(contour)
    side = (width + height) / 2
    box = cv2.boxPoints(((cx, cy), (side, side), angle))
    _draw_polygon(canvas, box.astype(np.int32).tolist(), color, thickness)


def _draw_rectangle(canvas: np.ndarray, contour: np.ndarray, color: BGR, thickness: int) -> None:
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    _draw_polygon(canvas, box.astype(np.int32).tolist(), color, thickness)


def _draw_triangle(
    canvas: np.ndarray, contour: np.ndarray, approx: np.ndarray, color: BGR, thickness: int
) -> None:
    """Reduce the stroke to its three extreme corners.

    ``minEnclosingTriangle`` finds the true corners even when the polygon
    approximation over- or under-shoots the vertex count.
    """
    try:
        area, triangle = cv2.minEnclosingTriangle(contour.astype(np.float32))
        if triangle is not None and area > 0:
            _draw_polygon(canvas, triangle.reshape(-1, 2).astype(np.int32).tolist(),
                          color, thickness)
            return
    except _RENDER_ERRORS as exc:
        logger.debug("minEnclosingTriangle failed: %s", exc)
    _draw_polygon(canvas, cv2.convexHull(approx).reshape(-1, 2).tolist(), color, thickness)


def _draw_hull(canvas: np.ndarray, contour: np.ndarray, color: BGR, thickness: int) -> None:
    """Last-resort rendering: the stroke's convex hull."""
    _draw_polygon(canvas, cv2.convexHull(contour).reshape(-1, 2).tolist(), color, thickness)


def _polygon_renderer(sides: int) -> Callable[[np.ndarray, np.ndarray, BGR, int], None]:
    """Build a renderer for a regular polygon with ``sides`` vertices."""

    def render(canvas: np.ndarray, contour: np.ndarray, color: BGR, thickness: int) -> None:
        center = _centroid(contour)
        offsets = contour.reshape(-1, 2).astype(float) - np.array(center)
        radii = np.linalg.norm(offsets, axis=1)
        circumradius = float(np.percentile(radii, 95))
        phase = _symmetry_phase(contour, center, folds=sides)

        points = [
            [int(center[0] + circumradius * math.cos(phase + i * 2 * math.pi / sides)),
             int(center[1] + circumradius * math.sin(phase + i * 2 * math.pi / sides))]
            for i in range(sides)
        ]
        _draw_polygon(canvas, points, color, thickness)

    return render


_RENDERERS: dict[str, Callable[[np.ndarray, np.ndarray, BGR, int], None]] = {
    "Heart": _draw_heart,
    "Star": _draw_star,
    "Line": _draw_line,
    "Circle": _draw_circle,
    "Ellipse": _draw_ellipse,
    "Square": _draw_square,
    "Rectangle": _draw_rectangle,
    "Pentagon": _polygon_renderer(5),
    "Hexagon": _polygon_renderer(6),
}


def draw_clean_shape(
    canvas: np.ndarray,
    shape_name: str,
    contour: np.ndarray,
    approx: np.ndarray,
    color: BGR,
    thickness: int,
) -> None:
    """Render ``shape_name`` onto ``canvas``, fitted to ``contour``."""
    try:
        if shape_name == "Triangle":
            _draw_triangle(canvas, contour, approx, color, thickness)
            return

        renderer = _RENDERERS.get(shape_name)
        if renderer is None:
            _draw_hull(canvas, contour, color, thickness)
            return
        renderer(canvas, contour, color, thickness)
    except _RENDER_ERRORS as exc:
        logger.warning("clean render of %s failed, falling back to hull: %s", shape_name, exc)
        _draw_hull(canvas, contour, color, thickness)


def render_mask(
    shape_name: str,
    contour: np.ndarray,
    approx: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterise the ideal shape as a solid single-channel mask.

    Used to measure how well the proposed shape covers what the user drew.
    """
    mask = np.zeros((height, width), np.uint8)
    draw_clean_shape(mask, shape_name, contour, approx, 255, cv2.FILLED)
    return mask
