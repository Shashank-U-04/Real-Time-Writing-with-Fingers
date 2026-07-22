"""GestureCanvas entry point.

Usage:
    python main.py [--camera N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gesture_canvas.app import run  # noqa: E402  (path set up above)


def main() -> int:
    parser = argparse.ArgumentParser(description="GestureCanvas - gesture controlled drawing")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    args = parser.parse_args()

    try:
        run(camera_index=args.camera)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
