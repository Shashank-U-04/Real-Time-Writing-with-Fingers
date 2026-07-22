"""Legacy entry point.

The application now lives in ``src/gesture_canvas``. This shim keeps the old
``python Deploy.py`` command working; new usage should prefer ``python main.py``.
"""

from main import main

if __name__ == "__main__":
    raise SystemExit(main())
