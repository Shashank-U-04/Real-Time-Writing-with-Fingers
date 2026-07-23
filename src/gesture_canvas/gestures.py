"""Gesture classification from a finger-up vector.

Named predicates rather than inline index comparisons, so the main loop reads as
intent and the priority order between overlapping gestures is stated in one
place. Order matters: a pinch and a two-finger selection share fingers, so the
more specific gesture must be tested first.
"""

from __future__ import annotations

from enum import Enum

from . import config

#: Indices into the finger-up vector.
THUMB, INDEX, MIDDLE, RING, PINKY = range(5)


class Gesture(Enum):
    """What the current hand pose means."""

    NONE = "none"
    RESIZE = "resize"
    ERASE = "erase"
    FILL = "fill"
    SELECT = "select"
    FIST = "fist"
    DRAW = "draw"


def _is_resize(f: list[int]) -> bool:
    """Thumb and index extended, nothing else — a measuring pinch."""
    return f[THUMB] == 1 and f[INDEX] == 1 and not any((f[MIDDLE], f[RING], f[PINKY]))


def _is_erase(f: list[int]) -> bool:
    """All four fingers extended — the original project's eraser.

    The thumb is deliberately ignored: the original finger vector did not
    include it, so a flat hand erases whether or not the thumb is tucked. That
    also means a fully relaxed open palm erases, which is exactly how the
    original behaved.
    """
    return all((f[INDEX], f[MIDDLE], f[RING], f[PINKY]))


def _is_fill(f: list[int]) -> bool:
    """Index, middle and ring extended — the paint bucket."""
    return f[INDEX] == 1 and f[MIDDLE] == 1 and f[RING] == 1 and f[PINKY] == 0


def _is_select(f: list[int]) -> bool:
    """Index and middle extended — move the cursor without drawing."""
    return f[INDEX] == 1 and f[MIDDLE] == 1 and f[RING] == 0


def _is_fist(f: list[int]) -> bool:
    return not any(f)


def _is_draw(f: list[int]) -> bool:
    """Index alone — put ink down."""
    return f[INDEX] == 1 and f[MIDDLE] == 0


#: Evaluated in order; the first match wins.
_RULES: tuple[tuple[Gesture, object], ...] = (
    (Gesture.RESIZE, _is_resize),
    (Gesture.ERASE, _is_erase),
    (Gesture.FILL, _is_fill),
    (Gesture.SELECT, _is_select),
    (Gesture.FIST, _is_fist),
    (Gesture.DRAW, _is_draw),
)


def classify(fingers: list[int], pinch_distance: float | None = None) -> Gesture:
    """Map a five-element finger-up vector onto a gesture.

    ``pinch_distance`` is the thumb-to-index tip distance in pixels. It
    disambiguates the one pose the original app and this one disagree on:
    thumb+index extended was *drawing* originally (the thumb was not tracked at
    all) but reads as a resize pinch here. Only a genuinely closed pinch resizes;
    a thumb merely resting open leaves you drawing, as it used to.

    Passing ``None`` means "distance unknown" and keeps the pose-only reading,
    which is what the gesture table describes.
    """
    if len(fingers) != 5:
        return Gesture.NONE

    for gesture, predicate in _RULES:
        if not predicate(fingers):
            continue
        if gesture is Gesture.RESIZE and not _pinch_engaged(pinch_distance):
            continue  # thumb is incidental — fall through to the drawing rule
        return gesture
    return Gesture.NONE


def _pinch_engaged(distance: float | None) -> bool:
    """True when the fingertips are close enough to mean a deliberate pinch."""
    return distance is None or distance <= config.PINCH_ENGAGE_DIST
