"""Gesture-driven paint bucket.

Filling a hand-drawn region has two problems a naive `cv2.floodFill` does not
solve:

1. **Leaks.** A stroke drawn by a fingertip rarely closes perfectly. Filling
   through a one-pixel gap floods the entire canvas, destroying the drawing. So
   the fill is first run as a dry run and rejected if it reaches the canvas edge.

2. **Halos.** Strokes are anti-aliased, so their edge pixels are a blend of ink
   and background. Filling only up to the hard edge leaves a pale outline
   between the fill and the stroke. The fill region is therefore dilated to pass
   *under* the stroke, and composited so the stroke stays fully opaque on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class FillResult:
    """Outcome of a fill attempt."""

    success: bool
    message: str

    @property
    def changed_canvas(self) -> bool:
        return self.success


#: Returned when the click is outside the drawable area or already the fill colour.
NO_OP = FillResult(False, "")


def _binary_ink_mask(canvas: np.ndarray, header_height: int) -> np.ndarray:
    """Ink as a hard 0/255 mask, with the header band sealed as a boundary.

    Thresholding discards the anti-aliased gradient so the fill sees crisp walls.
    The header band is marked solid so a shape drawn up against the toolbar is
    treated as closed by it, rather than leaking into the band and aborting.
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, config.INK_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask[: header_height + 1, :] = 255
    return mask


def _escaped(region: np.ndarray, header_height: int, marker: int) -> bool:
    """True if the fill reached a canvas edge, meaning the shape was not closed.

    Only the left, right and bottom edges are checked. The top is the sealed
    header band, so touching the row beneath it is expected for any shape the
    toolbar closes off, not evidence of a leak.
    """
    below_header = region[header_height + 1 :, :]
    return bool(
        np.any(below_header[-1, :] == marker)
        or np.any(below_header[:, 0] == marker)
        or np.any(below_header[:, -1] == marker)
    )


def flood_fill(
    canvas: np.ndarray,
    seed_x: int,
    seed_y: int,
    fill_color: config.BGR,
    header_height: int = config.HEADER_HEIGHT,
) -> tuple[np.ndarray, FillResult]:
    """Fill the enclosed region containing ``(seed_x, seed_y)``.

    Returns the new canvas and a result describing what happened. The input
    canvas is never modified in place, so a rejected fill costs the caller
    nothing and no undo entry is needed.
    """
    height, width = canvas.shape[:2]

    if not (0 <= seed_x < width and header_height < seed_y < height):
        return canvas, NO_OP
    if np.array_equal(canvas[seed_y, seed_x], np.array(fill_color, dtype=canvas.dtype)):
        return canvas, NO_OP

    ink = _binary_ink_mask(canvas, header_height)
    if ink[seed_y, seed_x] != 0:
        return canvas, FillResult(False, "Point inside a stroke — aim at empty space")

    # Dry run on the mask: 128 marks everything the fill would reach.
    marker = 128
    probe = ink.copy()
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(probe, flood_mask, (seed_x, seed_y), marker)

    if _escaped(probe, header_height, marker):
        return canvas, FillResult(False, "Fill aborted — shape is not closed")

    region = (probe == marker).astype(np.uint8)
    # Grow under the anti-aliased stroke edge so no pale halo is left behind.
    region = cv2.dilate(region, np.ones((3, 3), np.uint8), iterations=2)

    filled = _composite_under_strokes(canvas, region, fill_color)
    return filled, FillResult(True, "Area filled")


def _composite_under_strokes(
    canvas: np.ndarray, region: np.ndarray, fill_color: config.BGR
) -> np.ndarray:
    """Lay ``fill_color`` beneath existing ink inside ``region``.

    Coverage is read from each pixel's brightest channel: a solid stroke pixel
    is fully opaque and keeps its colour exactly, a partly-covered edge pixel
    lets a proportional amount of fill through, and empty pixels take the fill
    outright. Compositing this way keeps strokes crisp instead of tinting them.
    """
    canvas_f = canvas.astype(np.float32)
    coverage = np.max(canvas_f, axis=2, keepdims=True) / 255.0
    blended = np.clip(canvas_f + np.array(fill_color, np.float32) * (1.0 - coverage), 0, 255)

    result = canvas.copy()
    np.copyto(result, blended.astype(np.uint8), where=region[..., np.newaxis] == 1)
    return result
