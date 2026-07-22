"""Synthetic stroke fixtures.

Every fixture returns a list of ``(x, y)`` points shaped like something a user
would draw with a fingertip: sampled at irregular density, and deliberately not
closed, because a hand never returns exactly to its starting point.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

Point = tuple[int, int]

CENTER = (400, 400)


def _jitter(points: list[Point], amount: int = 3, seed: int = 7) -> list[Point]:
    """Add hand-tremor noise so tests exercise the smoothing path."""
    rng = random.Random(seed)
    return [(x + rng.randint(-amount, amount), y + rng.randint(-amount, amount)) for x, y in points]


def _interpolate(corners: list[Point], per_edge: int = 30) -> list[Point]:
    """Walk a polygon's outline, sampling evenly along each edge."""
    points: list[Point] = []
    for i in range(len(corners)):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % len(corners)]
        for step in range(per_edge):
            t = step / per_edge
            points.append((int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)))
    return points


@pytest.fixture
def circle_stroke() -> list[Point]:
    cx, cy = CENTER
    radius = 150
    # Stop short of 360 deg: a real stroke leaves a small gap.
    points = [
        (int(cx + radius * math.cos(math.radians(d))), int(cy + radius * math.sin(math.radians(d))))
        for d in range(0, 352, 3)
    ]
    return _jitter(points)


@pytest.fixture
def ellipse_stroke() -> list[Point]:
    cx, cy = CENTER
    points = [
        (int(cx + 210 * math.cos(math.radians(d))), int(cy + 95 * math.sin(math.radians(d))))
        for d in range(0, 352, 3)
    ]
    return _jitter(points)


@pytest.fixture
def square_stroke() -> list[Point]:
    return _jitter(_interpolate([(250, 250), (550, 250), (550, 550), (250, 550)], 40))


@pytest.fixture
def rectangle_stroke() -> list[Point]:
    return _jitter(_interpolate([(180, 300), (620, 300), (620, 490), (180, 490)], 40))


@pytest.fixture
def triangle_stroke() -> list[Point]:
    return _jitter(_interpolate([(400, 230), (580, 560), (220, 560)], 50))


@pytest.fixture
def pentagon_stroke() -> list[Point]:
    cx, cy = CENTER
    corners = [
        (int(cx + 170 * math.cos(math.radians(-90 + i * 72))),
         int(cy + 170 * math.sin(math.radians(-90 + i * 72))))
        for i in range(5)
    ]
    return _jitter(_interpolate(corners, 40), amount=2)


@pytest.fixture
def hexagon_stroke() -> list[Point]:
    cx, cy = CENTER
    corners = [
        (int(cx + 170 * math.cos(math.radians(i * 60))),
         int(cy + 170 * math.sin(math.radians(i * 60))))
        for i in range(6)
    ]
    return _jitter(_interpolate(corners, 40), amount=2)


@pytest.fixture
def star_stroke() -> list[Point]:
    cx, cy = CENTER
    corners: list[Point] = []
    for i in range(10):
        angle = i * math.pi / 5 - math.pi / 2
        radius = 190 if i % 2 == 0 else 75
        corners.append((int(cx + math.cos(angle) * radius), int(cy + math.sin(angle) * radius)))
    return _jitter(_interpolate(corners, 30), amount=2)


@pytest.fixture
def line_stroke() -> list[Point]:
    return _jitter([(150 + i * 5, 400) for i in range(120)], amount=2)


@pytest.fixture
def diagonal_line_stroke() -> list[Point]:
    """A diagonal line. Upright bounding boxes mis-measure this; oriented ones do not."""
    return _jitter([(150 + i * 4, 180 + i * 4) for i in range(130)], amount=2)


@pytest.fixture
def scribble_stroke() -> list[Point]:
    """Random noise that should classify as Freeform, not forced into a shape."""
    rng = random.Random(42)
    points: list[Point] = []
    x, y = CENTER
    for _ in range(200):
        x = max(60, min(740, x + rng.randint(-40, 40)))
        y = max(60, min(740, y + rng.randint(-40, 40)))
        points.append((x, y))
    return points


@pytest.fixture
def canvas_size() -> tuple[int, int]:
    """(height, width) large enough to hold every fixture with margin."""
    return 800, 800
