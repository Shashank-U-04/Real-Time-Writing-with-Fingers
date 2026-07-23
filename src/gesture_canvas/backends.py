"""MediaPipe hand-tracking backends.

MediaPipe removed the legacy ``solutions.hands`` API in release 0.10.30, and the
only releases available on Python 3.13 are newer than that. Two backends are
therefore provided behind one interface, chosen automatically:

* :class:`SolutionsBackend` — ``mp.solutions.hands``, on MediaPipe < 0.10.30.
* :class:`TasksBackend` — ``mp.tasks.vision.HandLandmarker``, the supported API
  going forward. Requires a model bundle on disk.

Both return landmarks in the same normalised form, so nothing downstream needs
to know which one is running.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from . import config

logger = logging.getLogger(__name__)

#: Google's published hand landmarker bundle for the Tasks API.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"

#: Guard against a truncated or error-page download being cached as a model.
_MIN_MODEL_BYTES = 1_000_000


class NormalisedLandmark(Protocol):
    """A landmark with coordinates in the 0.0-1.0 range."""

    x: float
    y: float


class HandBackend(Protocol):
    """What the hand detector needs from a tracking implementation."""

    def process(self, rgb: np.ndarray) -> tuple[Sequence[NormalisedLandmark] | None, str]:
        """Return the first hand's landmarks and its handedness label."""

    def close(self) -> None: ...


class SolutionsBackend:
    """Legacy ``mp.solutions.hands`` backend (MediaPipe < 0.10.30)."""

    name = "solutions"

    def __init__(
        self,
        static_mode: bool,
        max_hands: int,
        detection_confidence: float,
        tracking_confidence: float,
    ) -> None:
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, rgb: np.ndarray) -> tuple[Sequence[NormalisedLandmark] | None, str]:
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None, "Unknown"

        handedness = "Unknown"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label
        return results.multi_hand_landmarks[0].landmark, handedness

    def close(self) -> None:
        self._hands.close()


class TasksBackend:
    """Modern ``mp.tasks.vision.HandLandmarker`` backend."""

    name = "tasks"

    def __init__(
        self,
        static_mode: bool,
        max_hands: int,
        detection_confidence: float,
        tracking_confidence: float,
        model_path: Path,
    ) -> None:
        import mediapipe as mp

        self._mp = mp
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            # VIDEO mode carries tracking state between frames, so full palm
            # detection does not run every frame. Roughly doubles throughput
            # versus stateless IMAGE mode.
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        # VIDEO mode rejects a timestamp that does not advance, so a counter is
        # used rather than a wall clock: two frames processed inside the same
        # millisecond would otherwise make MediaPipe raise.
        self._timestamp_ms = 0

    def process(self, rgb: np.ndarray) -> tuple[Sequence[NormalisedLandmark] | None, str]:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 1
        result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        if not result.hand_landmarks:
            return None, "Unknown"

        handedness = "Unknown"
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name
        return result.hand_landmarks[0], handedness

    def close(self) -> None:
        self._landmarker.close()


# ── Model bundle management ──────────────────────────────────────────────────
def default_model_path() -> Path:
    """Where the Tasks model bundle is kept."""
    return Path(__file__).resolve().parents[2] / "assets" / MODEL_FILENAME


def ensure_model(path: Path | None = None, allow_download: bool = True) -> Path:
    """Return a path to the hand landmarker bundle, downloading it if needed.

    The download happens once. A partial file is written to a temporary name and
    only moved into place once complete, so an interrupted download can never
    leave a corrupt model that fails confusingly on the next run.
    """
    path = path or default_model_path()
    if path.is_file() and path.stat().st_size >= _MIN_MODEL_BYTES:
        return path

    if not allow_download:
        raise RuntimeError(
            f"Hand landmarker model not found at {path}.\n"
            f"Download it manually from:\n    {MODEL_URL}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    logger.info("Downloading hand tracking model (about 7 MB, one time only)...")

    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:  # noqa: S310
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Could not download the hand tracking model: {exc}\n"
            f"Download it manually from:\n    {MODEL_URL}\n"
            f"and save it as:\n    {path}"
        ) from exc

    if len(data) < _MIN_MODEL_BYTES:
        raise RuntimeError(
            f"Downloaded model looks truncated ({len(data)} bytes). "
            f"Retry, or fetch it manually from {MODEL_URL}"
        )

    temporary.write_bytes(data)
    os.replace(temporary, path)
    logger.info("Model saved to %s", path)
    return path


# ── Selection ────────────────────────────────────────────────────────────────
def supports_solutions() -> bool:
    """True when the installed MediaPipe still ships the legacy solutions API."""
    try:
        import mediapipe as mp
    except ImportError:
        return False
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")


def create_backend(
    static_mode: bool = False,
    max_hands: int = 1,
    detection_confidence: float = config.DETECTION_CONFIDENCE,
    tracking_confidence: float = config.TRACKING_CONFIDENCE,
    model_path: Path | None = None,
    allow_download: bool = True,
) -> HandBackend:
    """Build the best available backend for the installed MediaPipe."""
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "MediaPipe is not installed. Run: pip install -r requirements.txt"
        ) from exc

    if supports_solutions():
        logger.debug("using MediaPipe solutions backend")
        return SolutionsBackend(
            static_mode, max_hands, detection_confidence, tracking_confidence
        )

    if not hasattr(mp, "tasks"):
        raise RuntimeError(
            f"MediaPipe {getattr(mp, '__version__', '?')} provides neither the "
            "'solutions' nor the 'tasks' hand tracking API. "
            "Reinstall with: pip install -r requirements.txt"
        )

    logger.debug("using MediaPipe tasks backend")
    resolved = ensure_model(model_path, allow_download=allow_download)
    return TasksBackend(
        static_mode, max_hands, detection_confidence, tracking_confidence, resolved
    )
