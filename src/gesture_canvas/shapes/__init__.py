"""Shape recognition: raw fingertip stroke in, clean geometry out."""

from .classify import classify_shape
from .contour import resample_stroke, smooth_stroke, stroke_to_contour
from .render import draw_clean_shape

__all__ = [
    "classify_shape",
    "draw_clean_shape",
    "resample_stroke",
    "smooth_stroke",
    "stroke_to_contour",
]
