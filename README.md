# Real-Time Writing with Fingers ✍️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Gesture%20Recognition-orange?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

## 📌 Introduction

**Real-Time Writing with Fingers** is an interactive computer vision application that enables users to write, draw, and interact with a virtual canvas using simple hand gestures. By leveraging **OpenCV** and **MediaPipe**, this system detects hand landmarks in real-time, allowing for a touch-free drawing experience. 

It is designed for:
*   Online teaching & presentations 🎓
*   Quick digital note-taking 📝
*   Interactive demonstrations 🖌️

---

## 🚀 Key Features

*   **👆 Real-time Drawing:** Draw smoothly on the screen using your index finger.
*   **🖐️ Gesture Control:** 
    *   **Selection Mode:** Use two fingers (Index + Middle) to move the cursor without drawing.
    *   **Eraser Mode:** Use all fingers to erase parts of the canvas.
*   **🎨 Dynamic Customization:** Change brush colors and sizes actively from the on-screen header.
*   **⚡ Low Latency:** Optimized for real-time performance on standard CPUs.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core programming language. |
| **Computer Vision** | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) | Image processing and frame manipulation. |
| **Tracking** | ![MediaPipe](https://img.shields.io/badge/MediaPipe-000000?style=flat&logo=google&logoColor=white) | Hand landmark detection and tracking. |
| **Math** | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Array manipulation and coordinate calculations. |

---

## 🧩 Methodology

The system operates on a robust pipeline that processes video frames to detect hands and interpret gestures.

### System Workflow
1.  **Frame Capture:** Webcam feed is captured using OpenCV.
2.  **Hand Detection:** MediaPipe analyzes the frame to find hand landmarks.
3.  **Gesture Classification:** The system checks which fingers are up to switch between **Draw**, **Selection**, and **Erase** modes.
4.  **Canvas Update:** Lines are drawn on a separate canvas layer and merged with the original frame.

#### Flowchart
![System Flowchart](Flow%20Chart.png)

#### Hand Landmarks
The application relies on specific hand landmarks (Index tip: 8, Middle tip: 12) to identify gestures.
![Hand Landmarks](Hand_Landmarks.png)

---

## 📂 Project Structure

```bash
📦 Real-Time-Writing-with-Fingers
 ┣ 📂 NavBar                  # UI Assets for the header menu
 ┃ ┣ 📂 Colors               # Brush color selection icons
 ┃ ┣ 📂 Sizes                # Brush size icons
 ┃ ┗ 📂 Homepage             # Main header images
 ┣ 📜 Deploy.py               # Main application entry point
 ┣ 📜 HandTracking_GestureRecognition_Module.py # Hand detection logic
 ┣ 📜 Hand_Tracking.py        # Basic tracking utility
 ┣ 📜 Gesture_Recognition.py  # Recognition testing script
 ┣ 📜 Flow Chart.png          # System architecture diagram
 ┣ 📜 Hand_Landmarks.png      # Reference for MediaPipe landmarks
 ┗ 📜 README.md               # Project documentation
```

---

## 💻 Installation & Setup

### Prerequisites
Ensure you have **Python 3.x** installed.

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Shashank-U-04/Real-Time-Writing-with-Fingers.git
    cd Real-Time-Writing-with-Fingers
    ```

2.  **Install Dependencies**
    ```bash
    pip install opencv-python mediapipe numpy Pillow
    ```

3.  **Run the Application**
    ```bash
    python Deploy.py
    ```

---

## 🎮 Usage Guide

| Gesture | Fingers Up | Action |
| :--- | :--- | :--- |
| **Draw Mode** | ☝️ Index Only | Draw on the canvas. |
| **Selection Mode** | ✌️ Index + Middle | Move cursor / Select colors & sizes. |
| **Eraser Mode** | 🖐️ All Fingers | Erase content. |

> **Note:** Press `x` on your keyboard to exit the application.

---

## 🔜 Future Enhancements

- [ ] **Save Feature:** Button to save the current canvas as an image.
- [ ] **AI Shape Correction:** Auto-correct rough manual drawings into geometric shapes.
- [ ] **GUI Upgrade:** Porting the interface to PyQt or Tkinter for a native app feel.

---

## 📞 Support & Contact

*   **Name:** Shashank U
*   **GitHub:** [Shashank-U-04](https://github.com/Shashank-U-04)
*   **LinkedIn:** [Connect on LinkedIn](https://www.linkedin.com/in/shashank-u-016b54330/) 

---
<div align="center">
  Made with ❤️ by <a href="https://github.com/Shashank-U-04">Shashank U</a>
  <br>
  If you find this project useful, please give it a ⭐️ on GitHub!
</div>
