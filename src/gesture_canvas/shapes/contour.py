"""Turn a raw fingertip stroke into a contour the classifier can measure.

A stroke arrives with uneven point spacing (dense where the hand moved slowly)
and hand tremor riding on top. Both distort the geometric features the classifier
depends on, so the stroke is resampled to a uniform spacing and smoothed before
being rasterised.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import SHAPE_CONFIG

Point = tuple[int, int]


def resample_stroke(points: list[Point], spacing: int = SHAPE_CONFIG.resample_spacing) -> list[Point]:
    """Redistribute points at a uniform arc-length ``spacing``."""
    if len(points) < 2:
        return list(points)

    resampled: list[Point] = [tuple(points[0])]
    # Start one full spacing along: points[0] is already emitted, and sampling
    # from zero would duplicate it and leave a zero-length first gap.
    carry = float(spacing)

    for i in range(1, len(points)):
        p0 = np.array(points[i - 1], dtype=float)
        p1 = np.array(points[i], dtype=float)
        segment_length = float(np.linalg.norm(p1 - p0))
        if segment_length < 1e-6:
            continue

        distance = carry
        while distance < segment_length:
            t = distance / segment_length
            point = p0 + t * (p1 - p0)
            resampled.append((int(point[0]), int(point[1])))
            distance += spacing
        carry = distance - segment_length

    return resampled if len(resampled) >= 2 else list(points)


def smooth_stroke(points: list[Point], window: int = SHAPE_CONFIG.smooth_window) -> list[Point]:
    """Box-filter the stroke to suppress hand tremor."""
    if len(points) < window:
        return list(points)

    array = np.array(points, dtype=float)
    half = window // 2
    smoothed: list[Point] = []
    for i in range(len(array)):
        lo = max(0, i - half)
        hi = min(len(array), i + half + 1)
        mean = array[lo:hi].mean(axis=0)
        smoothed.append((int(mean[0]), int(mean[1])))
    return smoothed


def stroke_to_contour(
    points: list[Point], canvas_height: int, canvas_width: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Rasterise a stroke and extract its outer contour plus a polygon approximation.

    The stroke's interior is solid-filled before contour extraction. Without that,
    a hand-drawn loop that does not quite close reads as a thin ribbon, and every
    area-based feature (circularity, solidity) is computed against the ribbon
    rather than the shape the user meant to draw.

    Returns ``(None, None)`` when the stroke is too small or degenerate.
    """
    if len(points) < SHAPE_CONFIG.min_stroke_points:
        return None, None

    prepared = smooth_stroke(resample_stroke(points))
    array = np.array(prepared, dtype=np.int32)

    mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    thickness = max(14, int(min(canvas_width, canvas_height) * 0.015))
    cv2.fillPoly(mask, [array], 255)
    cv2.polylines(mask, [array], False, 255, thickness)

    # Close bridges gaps left by an unfinished loop; open removes stray specks.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < SHAPE_CONFIG.min_contour_area:
        return None, None

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, SHAPE_CONFIG.approx_epsilon_ratio * perimeter, True)
    return contour, approx
