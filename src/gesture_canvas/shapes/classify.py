"""Score-based shape classification.

Rather than a decision tree of thresholds, every candidate shape scores the same
feature set out of 100 and the highest wins. A tree commits early — one bad
corner count and a circle can never be reconsidered — whereas competing scores
let a strong signal on one axis outweigh a weak reading on another. If nothing
clears the confidence threshold the stroke is left as the user drew it.
"""

from __future__ import annotations

import numpy as np

from ..config import SHAPE_CONFIG
from .features import ShapeFeatures
from .verify import DEGENERATE_SHAPES, fit_iou

FREEFORM = "Freeform"


def _score_circle(f: ShapeFeatures) -> float:
    score = f.circularity * 45
    score += max(0.0, 1.0 - (f.ellipse_ratio - 1.0) / 0.5) * 20
    score += max(0.0, 1.0 - f.defects * 0.2) * 20
    score += f.solidity * 15
    if f.corners > 7:
        score += 10
    return min(score, 100.0)


def _score_ellipse(f: ShapeFeatures) -> float:
    score = f.circularity * 35
    score += min((f.ellipse_ratio - 1.0) / 0.5, 1.0) * 35
    score += max(0.0, 1.0 - f.defects * 0.2) * 20
    score += f.solidity * 10
    return min(score, 100.0)


def _score_triangle(f: ShapeFeatures) -> float:
    score = 0.0
    if f.corners == 3:
        score += 60
    elif f.corners == 4:
        score += 10
    score += (1.0 - f.circularity) * 25
    score += max(0.0, 1.0 - f.defects * 0.3) * 15
    return min(score, 100.0)


def _score_rectangle(f: ShapeFeatures) -> float:
    score = 0.0
    if f.corners == 4:
        score += 55
        # Only meaningful for a quadrilateral. Averaging the interior angles of a
        # 10-gon against 90 degrees produces a score that means nothing.
        if f.angles:
            error = float(np.mean([abs(a - 90) for a in f.angles]))
            score += max(0.0, 1.0 - error / 90) * 30
    score += (1.0 - f.circularity) * 15
    return min(score, 100.0)


def _score_square(f: ShapeFeatures) -> float:
    """Square shares Rectangle's base, then is rewarded or penalised on squareness.

    The adjustment must be able to go negative. If an off-square shape merely
    failed to earn a bonus, Square would tie Rectangle and win on tie-break —
    every wide rectangle would come back labelled "Square".
    """
    base = _score_rectangle(f)
    # Oriented aspect ratio, so a rotated square still reads as square.
    squareness = 1.0 - abs(f.min_aspect - 1.0) / 0.25
    adjustment = max(-1.0, min(squareness, 1.0)) * 20
    return min(base + adjustment, 100.0)


def _score_regular_polygon(f: ShapeFeatures, sides: int) -> float:
    score = max(0.0, 1.0 - abs(f.corners - sides) * 0.3) * 60
    return min(score + f.circularity * 20 + f.solidity * 20, 100.0)


def _score_star(f: ShapeFeatures) -> float:
    score = 0.0
    if f.defects == 5:
        score += 50
    elif 4 <= f.defects <= 6:
        score += 25
    # A star's points leave large gaps against its convex hull.
    if f.solidity < 0.60:
        score += 30
    elif f.solidity < 0.70:
        score += 15
    if f.corners >= 8:
        score += 20
    return min(score, 100.0)


def _score_heart(f: ShapeFeatures) -> float:
    score = f.top_notch * 35
    score += f.bottom_sharpness * 25
    score += f.symmetry * 20
    if f.defects == 1:
        score += 15
    elif f.defects == 0 or f.defects > 3:
        score -= 20
    score += (f.solidity - 0.6) * 20 if f.solidity > 0.6 else -10
    if f.circularity > 0.85:
        score -= 20
    return max(min(score, 100.0), 0.0)


def _score_line(f: ShapeFeatures) -> float:
    if f.min_aspect <= 4:
        return 0.0
    return min(50 + min((f.min_aspect - 4) * 5, 40), 100.0)


def score_all(features: ShapeFeatures) -> dict[str, float]:
    """Score every candidate shape against one feature set."""
    return {
        "Circle": _score_circle(features),
        "Ellipse": _score_ellipse(features),
        "Heart": _score_heart(features),
        "Square": _score_square(features),
        "Rectangle": _score_rectangle(features),
        "Triangle": _score_triangle(features),
        "Pentagon": _score_regular_polygon(features, 5),
        "Hexagon": _score_regular_polygon(features, 6),
        "Star": _score_star(features),
        "Line": _score_line(features),
    }


def classify_shape(contour: np.ndarray, approx: np.ndarray) -> tuple[str, float]:
    """Classify a contour.

    Candidates are scored, then the leader must also *look* like what it claims
    to be: the ideal shape is rendered and checked for overlap against the drawn
    stroke. Scoring alone lets an irregular scribble accumulate points across
    unrelated features without resembling anything.

    Returns ``(shape_name, confidence)`` where confidence is 0.0-1.0, or
    ``("Freeform", 0.0)`` when no candidate is convincing enough to justify
    replacing what the user drew.
    """
    features = ShapeFeatures.extract(contour, approx)
    scores = score_all(features)

    # Try candidates strongest-first: if the leader scores well but fits poorly,
    # the runner-up may be the shape actually drawn.
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    for name, score in ranked:
        if score < SHAPE_CONFIG.confidence_threshold:
            break

        if name in DEGENERATE_SHAPES:
            # No enclosed area to compare; the aspect-ratio test already decided.
            return name, round(score / 100.0, 2)

        overlap = fit_iou(name, contour, approx)
        if overlap < SHAPE_CONFIG.min_fit_iou:
            continue

        # Fold fit quality into the reported confidence so a loose match is
        # presented as one.
        confidence = (score / 100.0) * overlap
        if confidence * 100 < SHAPE_CONFIG.confidence_threshold:
            continue
        return name, round(confidence, 2)

    return FREEFORM, 0.0
