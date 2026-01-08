# Real-Time-Writing-With-Fingers

A computer vision application that allows users to write on their video feed using hand gestures in real-time. Ideal for online classes, meetings, and quick explanations.

## Features

- **Real-time Writing:** Use your index finger to draw on the screen.
- **Navigation:** Use index and middle fingers to navigate the menu.
- **Eraser:** Use four fingers to erase content.
- **Customization:** Change brush color and size dynamically.

## Prerequisites

Ensure you have Python installed. You will also need the following dependencies:

- `opencv-python`
- `mediapipe`
- `numpy`
- `Pillow`

## Installation

1. Install the required packages:
   ```bash
   pip install opencv-python mediapipe numpy Pillow
   ```

## Usage

1. Run the deployment script:
   ```bash
   python Deploy.py
   ```

2. **Gestures:**
   - **Draw:** Extend your **Index Finger**.
   - **Navigate/Select:** Extend **Index + Middle Fingers**.
   - **Erase:** Extend **Index + Middle + Ring + Little Fingers**.

3. Press `x` to exit the application.

## How It Works

The application uses **MediaPipe** for hand tracking and **OpenCV** for image processing.

1. **Hand Detection:** The webcam feed is processed to detect hand landmarks.
2. **Gesture Recognition:** The relative positions of fingertips are analyzed to determine the active gesture (Draw, Move, Erase).
3. **Drawing Logic:** Coordinates of the index finger are tracked to draw lines on a virtual canvas, which is then merged with the video feed using bitwise operations.

## Roadmap

- [ ] Implement a comprehensive GUI.
- [ ] Add support for more gestures.
- [ ] Optimize for lower latency on older hardware.
