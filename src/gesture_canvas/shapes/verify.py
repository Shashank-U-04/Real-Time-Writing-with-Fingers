"""Verify that a proposed shape actually fits the stroke the user drew.

Scoring alone is not enough. The feature scorers measure properties in isolation
— circularity, corner count, symmetry — so an irregular scribble can accumulate a
winning score without resembling any real shape. Before replacing a user's stroke
we render the proposed shape and check how much it overlaps what they drew.

Intersection-over-union is used because it penalises both directions of error:
a shape that spills outside the stroke and a shape that fails to cover it.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .render import render_mask

logger = logging.getLogger(__name__)

#: Shapes with no enclosed area cannot be compared by overlap.
DEGENERATE_SHAPES = frozenset({"Line"})

#: Padding around the stroke's bounding box, as a fraction of its size. Gives an
#: over-large proposal room to spill so the penalty is actually measured.
_PADDING_RATIO = 0.25


def _drawn_mask(contour: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, cv2.FILLED)
    return mask


def fit_iou(shape_name: str, contour: np.ndarray, approx: np.ndarray) -> float:
    """Overlap between the ideal ``shape_name`` and the drawn contour, 0.0-1.0.

    Rasterising happens in a padded region around the stroke rather than at full
    canvas size, so the cost does not depend on the camera resolution.
    """
    try:
        x, y, width, height = cv2.boundingRect(contour)
        pad_x = max(int(width * _PADDING_RATIO), 10)
        pad_y = max(int(height * _PADDING_RATIO), 10)
        roi_w = width + 2 * pad_x
        roi_h = height + 2 * pad_y

        # Translate into the local region so both masks share an origin.
        offset = np.array([[x - pad_x, y - pad_y]], dtype=np.int32)
        local_contour = contour.reshape(-1, 2) - offset
        local_approx = approx.reshape(-1, 2) - offset

        drawn = _drawn_mask(local_contour.reshape(-1, 1, 2), roi_h, roi_w)
        ideal = render_mask(
            shape_name,
            local_contour.reshape(-1, 1, 2),
            local_approx.reshape(-1, 1, 2),
            roi_h,
            roi_w,
        )

        intersection = int(np.count_nonzero(cv2.bitwise_and(drawn, ideal)))
        union = int(np.count_nonzero(cv2.bitwise_or(drawn, ideal)))
        if union == 0:
            return 0.0
        return intersection / union
    except (cv2.error, ValueError, IndexError) as exc:
        logger.debug("fit verification failed for %s: %s", shape_name, exc)
        # Fail open: a verification failure should not veto an otherwise
        # confident classification.
        return 1.0
