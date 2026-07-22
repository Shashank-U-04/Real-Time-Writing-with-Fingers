"""Tests for hand tracking and backend selection.

A fake backend supplies normalised landmarks, so the pixel conversion, finger
logic and handedness handling are all tested without MediaPipe or a camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from gesture_canvas import backends
from gesture_canvas.tracking import HandDetector, is_fist


@dataclass
class FakeLandmark:
    x: float
    y: float


class FakeBackend:
    """Returns a scripted hand pose."""

    name = "fake"

    def __init__(self, landmarks=None, handedness="Right") -> None:
        self.landmarks = landmarks
        self.handedness = handedness
        self.closed = False

    def process(self, rgb):
        return self.landmarks, self.handedness

    def close(self) -> None:
        self.closed = True


def open_hand(handedness="Right"):
    """21 landmarks with every finger extended (tips above their PIP joints)."""
    points = [FakeLandmark(0.5, 0.9) for _ in range(21)]
    for tip in (8, 12, 16, 20):
        points[tip] = FakeLandmark(0.5, 0.2)     # tip high on screen
        points[tip - 2] = FakeLandmark(0.5, 0.5)  # PIP joint lower
    # Right-hand thumb extends to the left in a mirrored frame.
    points[4] = FakeLandmark(0.30, 0.5)
    points[3] = FakeLandmark(0.40, 0.5)
    return points


def closed_hand():
    points = [FakeLandmark(0.5, 0.5) for _ in range(21)]
    for tip in (8, 12, 16, 20):
        points[tip] = FakeLandmark(0.5, 0.6)     # tip below its joint
        points[tip - 2] = FakeLandmark(0.5, 0.4)
    points[4] = FakeLandmark(0.55, 0.5)
    points[3] = FakeLandmark(0.45, 0.5)
    return points


@pytest.fixture
def frame():
    return np.zeros((720, 1280, 3), np.uint8)


def detector_with(landmarks, handedness="Right"):
    return HandDetector(backend=FakeBackend(landmarks, handedness))


# ── Landmark conversion ──────────────────────────────────────────────────────
def test_no_hand_yields_no_landmarks(frame):
    detector = detector_with(None)
    detector.find_hands(frame)

    landmarks, bbox = detector.find_position(frame)

    assert landmarks == []
    assert bbox is None


def test_normalised_coordinates_become_pixels(frame):
    detector = detector_with([FakeLandmark(0.5, 0.25)] * 21)
    detector.find_hands(frame)

    landmarks, _ = detector.find_position(frame)

    assert len(landmarks) == 21
    assert landmarks[0] == [0, 640, 180]


def test_bounding_box_spans_the_landmarks(frame):
    points = [FakeLandmark(0.5, 0.5) for _ in range(21)]
    points[0] = FakeLandmark(0.25, 0.25)
    points[20] = FakeLandmark(0.75, 0.75)
    detector = detector_with(points)
    detector.find_hands(frame)

    _, bbox = detector.find_position(frame)

    assert bbox == (320, 180, 960, 540)


def test_drawing_annotates_the_frame(frame):
    detector = detector_with(open_hand())

    detector.find_hands(frame, draw=True)

    assert frame.any(), "skeleton was not drawn"


def test_detection_without_draw_leaves_the_frame_clean(frame):
    detector = detector_with(open_hand())

    detector.find_hands(frame, draw=False)

    assert not frame.any()


# ── Finger state ─────────────────────────────────────────────────────────────
def test_finger_vector_has_five_entries(frame):
    detector = detector_with(open_hand())
    detector.find_hands(frame)
    detector.find_position(frame)

    assert len(detector.fingers_up()) == 5


def test_open_hand_reports_fingers_up(frame):
    detector = detector_with(open_hand())
    detector.find_hands(frame)
    detector.find_position(frame)

    assert detector.fingers_up() == [1, 1, 1, 1, 1]


def test_closed_hand_reports_a_fist(frame):
    detector = detector_with(closed_hand())
    detector.find_hands(frame)
    detector.find_position(frame)

    fingers = detector.fingers_up()

    assert fingers[1:] == [0, 0, 0, 0]
    assert is_fist([0, 0, 0, 0, 0])


def test_thumb_direction_flips_with_handedness(frame):
    """A mirrored selfie view reverses which way an extended thumb points."""
    points = open_hand()

    right = detector_with(points, "Right")
    right.find_hands(frame)
    right.find_position(frame)

    left = detector_with(points, "Left")
    left.find_hands(frame)
    left.find_position(frame)

    assert right.fingers_up()[0] != left.fingers_up()[0]


def test_finger_query_without_landmarks_is_all_down():
    detector = detector_with(None)
    assert detector.fingers_up() == [0, 0, 0, 0, 0]


# ── Geometry helpers ─────────────────────────────────────────────────────────
def test_distance_between_landmarks(frame):
    points = [FakeLandmark(0.5, 0.5) for _ in range(21)]
    points[4] = FakeLandmark(0.25, 0.5)   # x = 320
    points[8] = FakeLandmark(0.75, 0.5)   # x = 960
    detector = detector_with(points)
    detector.find_hands(frame)
    detector.find_position(frame)

    distance, geometry = detector.find_distance(4, 8)

    assert distance == pytest.approx(640, abs=1)
    assert geometry[4] == 640  # midpoint x


def test_distance_without_landmarks_is_zero():
    detector = detector_with(None)
    assert detector.find_distance(4, 8)[0] == 0.0


def test_right_angle_is_measured(frame):
    points = [FakeLandmark(0.5, 0.5) for _ in range(21)]
    points[0] = FakeLandmark(0.5, 0.25)
    points[1] = FakeLandmark(0.5, 0.5)
    points[2] = FakeLandmark(0.75, 0.5)
    detector = detector_with(points)
    detector.find_hands(frame)
    detector.find_position(frame)

    assert detector.find_angle(0, 1, 2) == pytest.approx(90, abs=1)


def test_angle_without_landmarks_is_zero():
    assert detector_with(None).find_angle(0, 1, 2) == 0.0


# ── Lifecycle ────────────────────────────────────────────────────────────────
def test_close_releases_the_backend():
    backend = FakeBackend(None)
    HandDetector(backend=backend).close()
    assert backend.closed


def test_backend_name_is_reported():
    assert detector_with(None).backend_name == "fake"


# ── Backend selection ────────────────────────────────────────────────────────
def test_missing_mediapipe_gives_an_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fail_on_mediapipe(name, *args, **kwargs):
        if name == "mediapipe":
            raise ImportError("no mediapipe")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_on_mediapipe)

    with pytest.raises(RuntimeError, match="pip install"):
        backends.create_backend()


def test_solutions_support_is_detected_without_raising():
    assert isinstance(backends.supports_solutions(), bool)


def test_missing_model_without_download_explains_where_to_get_it(tmp_path):
    with pytest.raises(RuntimeError, match="Download it manually"):
        backends.ensure_model(tmp_path / "absent.task", allow_download=False)


def test_existing_model_is_reused(tmp_path):
    """A cached model must not trigger another download."""
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"x" * 2_000_000)

    assert backends.ensure_model(model, allow_download=False) == model


def test_truncated_model_is_not_accepted(tmp_path):
    """A partial download must be rejected rather than loaded and failing oddly."""
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"404 not found")

    with pytest.raises(RuntimeError):
        backends.ensure_model(model, allow_download=False)


def test_default_model_path_is_in_assets():
    assert backends.default_model_path().name == backends.MODEL_FILENAME
    assert backends.default_model_path().parent.name == "assets"
