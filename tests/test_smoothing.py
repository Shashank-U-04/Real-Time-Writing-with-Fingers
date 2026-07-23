"""Tests for adaptive cursor smoothing.

Smoothing is disabled by default (see `config.SMOOTHING_ENABLED`), so the tests
that exercise the filter maths opt in explicitly. `test_passthrough_*` cover the
default, which is what actually reaches the canvas.
"""

from __future__ import annotations

import math

from gesture_canvas import config
from gesture_canvas.smoothing import CursorSmoother


def filtering() -> CursorSmoother:
    """A smoother with the filter switched on."""
    return CursorSmoother(enabled=True)


# ── Passthrough (the default) ────────────────────────────────────────────────
def test_smoothing_is_off_by_default():
    """This is a handwriting tool; lag matters more than jitter."""
    assert config.SMOOTHING_ENABLED is False
    assert CursorSmoother().enabled is False


def test_passthrough_returns_the_raw_position():
    smoother = CursorSmoother()
    for point in [(100, 200), (400, 380), (401, 379), (900, 120)]:
        assert smoother.update(*point) == point


def test_passthrough_never_lags_behind_the_finger():
    """The original drew from the raw fingertip; ink must not trail it."""
    smoother = CursorSmoother()
    path = [(60 + i * 7, 300 + int(45 * math.sin(i / 26 * 2 * math.pi)))
            for i in range(140)]

    lags = [math.hypot(raw[0] - out[0], raw[1] - out[1])
            for raw, out in ((p, smoother.update(*p)) for p in path)]

    assert max(lags) == 0


def test_passthrough_preserves_stroke_amplitude():
    """Loops must keep their full height or letters collapse."""
    smoother = CursorSmoother()
    path = [(60 + i * 7, 300 + int(45 * math.sin(i / 26 * 2 * math.pi)))
            for i in range(140)]

    out_y = [smoother.update(*p)[1] for p in path]

    assert max(out_y) - min(out_y) == max(y for _, y in path) - min(y for _, y in path)


def test_passthrough_output_is_integral():
    x, y = CursorSmoother().update(37.6, 91.4)
    assert isinstance(x, int) and isinstance(y, int)


# ── Filter maths (opt-in) ────────────────────────────────────────────────────
def test_first_sample_passes_through_unchanged():
    """There is no history to smooth against, so the cursor must not lag."""
    smoother = filtering()
    assert smoother.update(100, 200) == (100, 200)


def test_output_converges_on_a_held_position():
    smoother = filtering()
    smoother.update(0, 0)

    for _ in range(200):
        result = smoother.update(500, 300)

    assert abs(result[0] - 500) <= 1
    assert abs(result[1] - 300) <= 1


def test_smoothing_suppresses_jitter():
    """A hand held still still jitters; the output should barely move."""
    smoother = filtering()
    smoother.update(400, 400)

    positions = []
    for i in range(40):
        offset = 3 if i % 2 == 0 else -3
        positions.append(smoother.update(400 + offset, 400 + offset))

    xs = [x for x, _ in positions[10:]]
    assert max(xs) - min(xs) <= 4, "jitter passed straight through"


def test_fast_motion_uses_a_higher_blend_factor():
    """Slow strokes are smoothed heavily; fast ones must stay responsive."""
    smoother = filtering()

    slow = smoother.alpha_for_speed(1.0)
    fast = smoother.alpha_for_speed(200.0)

    assert fast > slow
    assert slow >= smoother.alpha_slow
    assert fast <= smoother.alpha_fast


def test_blend_factor_is_bounded():
    smoother = filtering()
    for distance in (0.0, 5.0, 40.0, 1000.0):
        alpha = smoother.alpha_for_speed(distance)
        assert smoother.alpha_slow <= alpha <= smoother.alpha_fast


def test_fast_motion_tracks_more_closely_than_slow_motion():
    """A quick flick should land nearer the target than a crawl would."""
    fast_smoother = filtering()
    fast_smoother.update(0, 0)
    fast_result = fast_smoother.update(300, 0)

    slow_smoother = filtering()
    slow_smoother.update(0, 0)
    slow_smoother.update(1, 0)  # establish a slow baseline
    slow_result = slow_smoother.update(4, 0)

    fast_fraction = fast_result[0] / 300
    slow_fraction = (slow_result[0] - 1) / 3
    assert fast_fraction > slow_fraction


def test_reset_clears_history():
    smoother = filtering()
    smoother.update(10, 10)
    smoother.update(20, 20)

    smoother.reset()

    assert smoother.update(700, 500) == (700, 500)


def test_output_is_integral():
    smoother = filtering()
    smoother.update(0, 0)
    x, y = smoother.update(37, 91)
    assert isinstance(x, int) and isinstance(y, int)
