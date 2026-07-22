"""Geometric features extracted from a contour.

Each function reduces a contour to a single scalar the classifier can score
against. They are deliberately independent so a bad reading from one (a failed
ellipse fit, say) degrades a score rather than aborting classification.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Errors OpenCV raises on degenerate contours: too few points, singular fits.
_GEOMETRY_ERRORS = (cv2.error, ValueError, ZeroDivisionError, IndexError)


def circularity(contour: np.ndarray) -> float:
    """4*pi*area / perimeter^2. 1.0 for a perfect circle, lower for anything else."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    return 4 * math.pi * area / (perimeter * perimeter + 1e-6)


def solidity(contour: np.ndarray) -> float:
    """Contour area over its convex hull area. Low for spiky shapes like stars."""
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    return area / (hull_area + 1e-6)


def ellipse_ratio(contour: np.ndarray) -> float:
    """Major/minor axis ratio of the best-fit ellipse. 1.0 means circular."""
    if len(contour) < 5:
        return 1.0
    try:
        _, (major, minor), _ = cv2.fitEllipse(contour)
        return max(major, minor) / max(min(major, minor), 1e-6)
    except _GEOMETRY_ERRORS as exc:
        logger.debug("ellipse fit failed: %s", exc)
        return 1.0


def defect_count(contour: np.ndarray, min_depth_ratio: float = 0.10) -> int:
    """Number of significant convexity defects — the notches a star or heart has."""
    try:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 3:
            return 0
        defects = cv2.convexityDefects(contour, hull_indices)
        if defects is None or defects.size == 0:
            return 0

        # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4). Flatten either
        # into rows of [start, end, farthest, depth] so both behave the same.
        depths = defects.reshape(-1, 4)[:, 3]

        _, _, width, height = cv2.boundingRect(contour)
        # Depth is reported in fixed-point units of 1/256 px.
        threshold = min(width, height) * min_depth_ratio * 256
        return int(np.sum(depths > threshold))
    except _GEOMETRY_ERRORS as exc:
        logger.debug("defect analysis failed: %s", exc)
        return 0


def bilateral_symmetry(contour: np.ndarray) -> float:
    """How closely the left and right halves mirror each other, 0.0 to 1.0."""
    try:
        points = contour.reshape(-1, 2).astype(float)
        center_x = points[:, 0].mean()
        left = points[points[:, 0] < center_x]
        right = points[points[:, 0] >= center_x]
        if len(left) < 3 or len(right) < 3:
            return 0.5

        left_span = left[:, 1].max() - left[:, 1].min()
        right_span = right[:, 1].max() - right[:, 1].min()
        difference = abs(left_span - right_span) / (max(left_span, right_span) + 1e-6)
        return max(0.0, 1.0 - difference)
    except _GEOMETRY_ERRORS as exc:
        logger.debug("symmetry analysis failed: %s", exc)
        return 0.5


def bottom_point_sharpness(contour: np.ndarray) -> float:
    """How pointed the bottom is. 1.0 for a heart's tip, ~0 for a flat base."""
    try:
        points = contour.reshape(-1, 2).astype(float)
        _, _, width, height = cv2.boundingRect(contour)
        lowest_y = points[:, 1].max()
        bottom_band = points[points[:, 1] > lowest_y - height * 0.1]
        if len(bottom_band) < 1:
            return 0.0
        spread = bottom_band[:, 0].max() - bottom_band[:, 0].min()
        return 1.0 - min(spread / (width + 1e-6), 1.0)
    except _GEOMETRY_ERRORS as exc:
        logger.debug("sharpness analysis failed: %s", exc)
        return 0.0


def top_notch_depth(contour: np.ndarray) -> float:
    """Depth of a central dip along the top edge — a heart's cleft."""
    try:
        points = contour.reshape(-1, 2).astype(float)
        x, y, width, height = cv2.boundingRect(contour)
        center_x = x + width / 2

        top_band = points[points[:, 1] < y + height * 0.25]
        if len(top_band) < 3:
            return 0.0
        near_center = top_band[np.abs(top_band[:, 0] - center_x) < width * 0.2]
        if len(near_center) < 1:
            return 0.0

        notch_y = near_center[:, 1].max()
        top_y = points[:, 1].min()
        depth = (notch_y - top_y) / (height + 1e-6)
        # Scaled so a shallow but real cleft still reads as a strong signal.
        return min(depth * 4, 1.0)
    except _GEOMETRY_ERRORS as exc:
        logger.debug("notch analysis failed: %s", exc)
        return 0.0


def min_area_aspect(contour: np.ndarray) -> float:
    """Aspect ratio of the *oriented* bounding box.

    An upright bounding box reports a 45-degree line as nearly square, which is
    why orientation-independent sizing matters for line and square detection.
    """
    try:
        (_, _), (width, height), _ = cv2.minAreaRect(contour)
        return max(width, height) / max(min(width, height), 1e-6)
    except _GEOMETRY_ERRORS as exc:
        logger.debug("min-area-rect failed: %s", exc)
        return 1.0


def interior_angles(polygon: np.ndarray) -> list[float]:
    """Interior angles in degrees at each vertex of an approximated polygon."""
    points = polygon.reshape(-1, 2).astype(float)
    count = len(points)
    if count < 3:
        return []

    angles: list[float] = []
    for i in range(count):
        v1 = points[i - 1] - points[i]
        v2 = points[(i + 1) % count] - points[i]
        denominator = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
        cosine = float(np.dot(v1, v2) / denominator)
        angles.append(math.degrees(math.acos(np.clip(cosine, -1.0, 1.0))))
    return angles


@dataclass(frozen=True)
class ShapeFeatures:
    """All features for one contour, computed once and shared by every scorer."""

    corners: int
    circularity: float
    solidity: float
    ellipse_ratio: float
    defects: int
    symmetry: float
    bottom_sharpness: float
    top_notch: float
    min_aspect: float
    angles: tuple[float, ...]

    @classmethod
    def extract(cls, contour: np.ndarray, approx: np.ndarray) -> "ShapeFeatures":
        return cls(
            corners=len(approx),
            circularity=circularity(contour),
            solidity=solidity(contour),
            ellipse_ratio=ellipse_ratio(contour),
            defects=defect_count(contour),
            symmetry=bilateral_symmetry(contour),
            bottom_sharpness=bottom_point_sharpness(contour),
            top_notch=top_notch_depth(contour),
            min_aspect=min_area_aspect(contour),
            angles=tuple(interior_angles(approx)),
        )
