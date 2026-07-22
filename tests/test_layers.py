"""Tests for layered rendering, undo history and the snap animation."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from gesture_canvas.layers import LayerManager, SnapAnimation, UndoStack

HEIGHT, WIDTH = 400, 600
RED = (0, 0, 200)
GREEN = (0, 200, 0)


@pytest.fixture
def layers():
    return LayerManager(HEIGHT, WIDTH)


# ── Drawing ──────────────────────────────────────────────────────────────────
def test_stroke_marks_the_base_layer(layers):
    layers.draw_stroke((100, 100), (200, 100), RED, 10)
    assert tuple(layers.base[100, 150]) == RED


def test_stroke_has_no_gaps(layers):
    """Circle stamping must leave a continuous line, not a dotted one."""
    layers.draw_stroke((50, 200), (550, 200), RED, 8)

    row = layers.base[200, 50:551]
    assert row.any(axis=1).all(), "stroke has gaps along its length"


def test_stroke_respects_thickness(layers):
    layers.draw_stroke((300, 200), (300, 200), RED, 20)

    column = layers.base[:, 300]
    inked = np.flatnonzero(column.any(axis=1))
    assert 16 <= (inked.max() - inked.min() + 1) <= 24


def test_erase_removes_ink(layers):
    layers.draw_stroke((100, 100), (300, 100), RED, 20)
    layers.erase_stroke((100, 100), (300, 100), 40)
    assert not layers.base[100, 200].any()


def test_clear_empties_everything(layers):
    layers.draw_stroke((10, 10), (100, 100), RED, 10)
    layers.start_preview(lambda c: cv2.circle(c, (200, 200), 40, GREEN, 5))

    layers.clear()

    assert not layers.base.any()
    assert layers.preview is None


# ── Compositing ──────────────────────────────────────────────────────────────
def test_composite_without_preview_matches_base(layers):
    layers.draw_stroke((10, 10), (100, 100), RED, 10)
    assert np.array_equal(layers.composite(), layers.base)


def test_composite_shows_preview_over_base(layers):
    layers.draw_stroke((100, 200), (500, 200), RED, 10)
    layers.start_preview(lambda c: cv2.line(c, (100, 200), (500, 200), GREEN, 10))

    result = layers.composite()

    assert tuple(result[200, 300]) == GREEN


def test_preview_does_not_touch_the_base(layers):
    layers.draw_stroke((100, 200), (500, 200), RED, 10)
    before = layers.base.copy()

    layers.start_preview(lambda c: cv2.circle(c, (300, 200), 80, GREEN, cv2.FILLED))

    assert np.array_equal(layers.base, before)


def test_cancelling_a_preview_restores_the_original_view(layers):
    layers.draw_stroke((100, 200), (500, 200), RED, 10)
    original = layers.composite()

    layers.start_preview(lambda c: cv2.circle(c, (300, 200), 80, GREEN, cv2.FILLED))
    layers.cancel_preview()

    assert np.array_equal(layers.composite(), original)


def test_commit_replaces_the_raw_stroke_with_the_clean_shape(layers):
    # Arrange - a messy diagonal stroke, to be replaced by a clean horizontal line
    stroke = [(100, 150), (200, 160), (300, 140), (400, 155)]
    for a, b in zip(stroke, stroke[1:]):
        layers.draw_stroke(a, b, RED, 8)
    layers.start_preview(lambda c: cv2.line(c, (100, 300), (400, 300), GREEN, 8))

    # Act
    layers.commit_preview(stroke, 8)

    # Assert - the clean shape is baked in and the raw stroke is gone
    assert tuple(layers.base[300, 250]) == GREEN
    assert not layers.base[150:165, 100:400].any(), "raw stroke was not erased"
    assert layers.preview is None


def test_commit_without_a_preview_is_harmless(layers):
    layers.draw_stroke((10, 10), (100, 100), RED, 10)
    before = layers.base.copy()

    layers.commit_preview([(10, 10), (100, 100)], 10)

    assert np.array_equal(layers.base, before)


# ── Undo ─────────────────────────────────────────────────────────────────────
def test_undo_restores_the_previous_canvas(layers):
    stack = UndoStack()
    stack.push(layers.snapshot())
    layers.draw_stroke((100, 100), (300, 100), RED, 20)

    layers.restore(stack.pop())

    assert not layers.base.any()


def test_undo_stack_is_bounded():
    stack = UndoStack(depth=3)
    for i in range(10):
        stack.push(np.full((4, 4, 3), i, np.uint8))

    assert len(stack) == 3
    assert stack.pop()[0, 0, 0] == 9


def test_popping_an_empty_stack_returns_none():
    assert UndoStack().pop() is None


def test_snapshots_are_independent_copies(layers):
    stack = UndoStack()
    stack.push(layers.snapshot())

    layers.draw_stroke((100, 100), (300, 100), RED, 20)

    assert not stack.pop().any(), "snapshot aliased the live canvas"


def test_restore_discards_any_pending_preview(layers):
    stack = UndoStack()
    stack.push(layers.snapshot())
    layers.start_preview(lambda c: cv2.circle(c, (200, 200), 40, GREEN, 5))

    layers.restore(stack.pop())

    assert layers.preview is None


# ── Animation ────────────────────────────────────────────────────────────────
def test_inactive_animation_returns_the_fallback():
    anim = SnapAnimation()
    fallback = np.full((10, 10, 3), 7, np.uint8)
    assert np.array_equal(anim.update(fallback), fallback)


def test_animation_blends_between_the_two_frames():
    before = np.zeros((10, 10, 3), np.uint8)
    after = np.full((10, 10, 3), 255, np.uint8)
    anim = SnapAnimation(duration_ms=400)

    anim.start(before, after)
    time.sleep(0.15)  # roughly a third of the way through
    mid = anim.update(before)

    assert anim.active
    assert 0 < int(mid[5, 5, 0]) < 255


def test_animation_finishes_on_the_target_frame():
    before = np.zeros((10, 10, 3), np.uint8)
    after = np.full((10, 10, 3), 255, np.uint8)
    anim = SnapAnimation(duration_ms=10)

    anim.start(before, after)
    time.sleep(0.05)
    result = anim.update(before)

    assert anim.finished
    assert np.array_equal(result, after)
