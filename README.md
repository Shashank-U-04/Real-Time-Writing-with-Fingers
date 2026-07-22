# GestureCanvas 🎨

**AI-powered gesture-controlled smart drawing system**

![Python](https://img.shields.io/badge/Python-3.10%20--%203.13-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange?style=for-the-badge&logo=google&logoColor=white)
![Tests](https://img.shields.io/badge/tests-216%20passing-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📌 Introduction

**GestureCanvas** turns your hand into a drawing tool. A webcam tracks 21 hand
landmarks in real time, and finger poses become drawing commands — no mouse, no
stylus, no touchscreen.

Beyond freehand drawing it adds **AI shape recognition** (rough sketches snap to
clean geometry), a **leak-proof smart fill**, and a **layered canvas** that keeps
every AI suggestion reversible until you accept it.

Built for:
* Online teaching and presentations 🎓
* Interactive demos and kiosks 🖥️
* Touch-free digital note-taking 📝

---

## 🚀 Features

### ✋ Real-time hand tracking
21 MediaPipe landmarks, single-hand control, handedness-aware thumb detection,
and speed-adaptive smoothing that removes jitter without adding lag.

### 🎨 Smart drawing
7 colours, continuously variable brush (4–50 px) and eraser (20–120 px) sized by
pinching, and round-capped anti-aliased strokes with uniform width through
direction changes.

### 🧠 AI shape recognition
Draw roughly; get clean geometry. Ten shape classes are scored against a shared
feature set, and the winner must additionally **fit** the stroke before it is
accepted — so a scribble stays a scribble instead of being forced into a circle.

| Supported | | |
|---|---|---|
| Circle | Ellipse | Line |
| Square | Rectangle | Triangle |
| Pentagon | Hexagon | Star |
| Heart | | |

Shapes are fitted to your stroke's real centre, size and **rotation** — a tilted
square snaps to a tilted square.

### 🪣 Smart fill
A gesture-driven paint bucket with a dry-run leak check: if your outline has a
gap, the fill is refused and reported rather than flooding the canvas. Fill
slides *under* anti-aliased strokes, so there is no pale halo at the boundary.

### 🧱 Layered canvas
Separate drawing and AI-preview layers, a 25-step undo history, and an animated
crossfade when a shape snaps — all non-destructive until confirmed.

### ⚡ Stability
Hover debounce, click cooldown, resize debounce, and an 8-frame fist
confirmation, so tracking jitter never triggers an action you did not intend.

---

## 🤏 Gesture guide

| Gesture | Fingers | Action |
|:---|:---|:---|
| ☝️ Index only | `[_,1,0,_,_]` | **Draw** (or erase, if the eraser is active) |
| ✌️ Index + Middle | `[_,1,1,0,_]` | **Select** — tap toolbar buttons and colours |
| 🤏 Thumb + Index | `[1,1,0,0,0]` | **Resize** the brush or eraser, with a live preview |
| 🖐️ Index + Middle + Ring | `[_,1,1,1,0]` | **Fill** the enclosed area under your fingertip |
| ✊ Fist (hold) | `[0,0,0,0,0]` | **Confirm** an AI shape snap |
| 🖐️ Open palm | `[_,1,1,1,1]` | Neutral — does nothing |

**Toolbar:** `colours ×7 │ brush │ eraser │ undo │ clear │ save │ ai │ settings`

**Keyboard:** `q`/`Esc` quit · `z` undo · `s` save · `c` clear

### Using the AI snap
1. Tap **AI** in the toolbar — the badge reads `DRAW`
2. Draw your shape with one finger, then lower it — the badge reads `FIST`
3. Hold a fist for about 8 frames — the shape snaps, with a confidence readout

Tap **AI** again at any point to cancel.

---

## 🛠️ Tech stack

| Component | Technology |
|:---|:---|
| Language | Python 3.10+ |
| Computer vision | OpenCV |
| Hand tracking | MediaPipe |
| Numerics | NumPy |
| Testing | pytest, pytest-cov |

---

## 💻 Installation

```bash
git clone https://github.com/Shashank-U-04/Real-Time-Writing-with-Fingers.git
cd Real-Time-Writing-with-Fingers
pip install -r requirements.txt
python main.py
```

Optional flags:

```bash
python main.py --camera 1     # use a different webcam
```

### MediaPipe compatibility
MediaPipe removed the legacy `solutions.hands` API in release **0.10.30**, and
the only releases available on **Python 3.13** are newer than that. GestureCanvas
supports both APIs and picks one automatically:

| Your environment | Backend used | Setup |
|:---|:---|:---|
| MediaPipe < 0.10.30 (Python ≤ 3.12) | `solutions.hands` | nothing extra |
| MediaPipe ≥ 0.10.30 (incl. Python 3.13) | `tasks.HandLandmarker` | model fetched once, automatically |

On the Tasks backend the hand landmarker model (~7 MB) is downloaded to
`assets/hand_landmarker.task` on first run and reused thereafter. To install it
ahead of time, or if the machine is offline:

```bash
curl -o assets/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Measured steady-state throughput on the Tasks backend is **~30 FPS** on CPU
(webcam-capped). The first few seconds after launch are slower while the model
warms up.

### Optional: custom toolbar artwork
Drop an image at `assets/header.png` to use it as the toolbar background.
Interactive highlights are always drawn on top, so the app looks and behaves
correctly with or without it.

---

## 🧩 How it works

### Pipeline
1. **Capture** — frame grabbed, resized to 1280×720, mirrored
2. **Track** — MediaPipe returns 21 landmarks
3. **Classify gesture** — the finger-up vector maps to one intent
4. **Dispatch** — draw, select, resize, fill, or confirm
5. **Composite** — canvas keyed over the camera feed, overlays drawn on top

### Shape recognition
```
stroke → resample → smooth → rasterise (solid-filled) → contour
       → features → score 10 candidates → verify fit (IoU) → clean render
```

The stroke's interior is **solid-filled** before analysis. Without this, a
hand-drawn loop that does not quite close reads as a thin ribbon, and every
area-based measurement describes the ribbon rather than the intended shape.

The **fit check** is what separates this from pure scoring: the winning shape is
rendered and compared against the drawn stroke by intersection-over-union. A
candidate that scores well but does not overlap the stroke is rejected, and the
runner-up is tried instead.

#### Reference diagrams
![System Flowchart](Flow%20Chart.png)
![Hand Landmarks](Hand_Landmarks.png)

---

## 📂 Project structure

```
GestureCanvas/
├── main.py                     # entry point
├── Deploy.py                   # legacy shim → main.py
├── requirements.txt
├── assets/
│   ├── header.png              # optional toolbar artwork
│   └── hand_landmarker.task    # auto-downloaded, gitignored
├── src/gesture_canvas/
│   ├── config.py               # all constants; toolbar zones
│   ├── state.py                # AppState + AI state machine
│   ├── tracking.py             # hand detector, landmarks → fingers
│   ├── backends.py             # MediaPipe solutions / tasks backends
│   ├── gestures.py             # finger vector → intent
│   ├── smoothing.py            # speed-adaptive cursor filter
│   ├── layers.py               # layers, undo, snap animation
│   ├── fill.py                 # leak-proof flood fill
│   ├── toolbar.py              # header render + hit-testing
│   ├── overlay.py              # on-screen feedback
│   ├── app.py                  # camera loop + dispatch
│   └── shapes/
│       ├── contour.py          # stroke → contour
│       ├── features.py         # geometric features
│       ├── classify.py         # scoring + decision
│       ├── verify.py           # IoU fit check
│       └── render.py           # clean shape drawing
└── tests/                      # 216 tests, no webcam required
```

---

## 🧪 Testing

```bash
pip install pytest pytest-cov
pytest                                              # 216 tests, ~2s
pytest --cov=gesture_canvas --cov-report=term-missing
```

Tests run entirely on synthetic strokes and numpy canvases — **no webcam
needed** — MediaPipe is stubbed behind the backend interface. Coverage is 88%
overall, and 82–100% across every module except `backends.py`, whose branches
depend on which MediaPipe release is installed.

---

## 🔜 Future enhancements

- [ ] OCR for handwritten text recognition
- [ ] Mathematical expression recognition
- [ ] Multi-hand collaborative drawing
- [ ] SVG / PDF export
- [ ] User-customisable gesture bindings

---

## 📞 Contact

* **Name:** Shashank U
* **GitHub:** [Shashank-U-04](https://github.com/Shashank-U-04)
* **LinkedIn:** [Connect](https://www.linkedin.com/in/shashank-u-016b54330/)

---

## 🙏 Acknowledgements

Feature direction inspired by
[GestureCanvas-AI](https://github.com/syedshayaanuddin/GestureCanvas-AI) by
Syed Shayaan Uddin.

---

<div align="center">
  Made with ❤️ by <a href="https://github.com/Shashank-U-04">Shashank U</a>
  <br>
  If you find this project useful, please give it a ⭐️ on GitHub!
</div>
