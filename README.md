# HADB – Hybrid Adaptive Driving Beam System

This project implements an Adaptive Driving Beam (ADB) system using computer vision. It intelligently dims sections of a headlight beam to avoid glaring other drivers, based on real-time video analysis.

## Features
- **Real-time Video Processing:** Ingests video from a file or webcam.
- **Dual-Detection System:**
    - **Brightness-based:** Fast detection of bright light sources (headlights, taillights).
    - **AI-based:** Slower, more accurate detection of vehicles using a YOLOv8 object detection model.
- **Grid-based Beam Control:** Divides the view into a grid and dims only the cells containing detected vehicles or bright lights.
- **Modular & Configurable:** Key parameters like detection sensitivity, grid size, and video source are managed in `config.json`.
- **Hardware Abstraction:** Includes a mock hardware layer for development and a base class for integration with real hardware (like Raspberry Pi GPIO).

## Tech Stack
- Python
- OpenCV for image processing
- NumPy for numerical operations
- Ultralytics YOLOv8 for object detection
- PyTorch as the backend for the YOLO model

## Setup Instructions
1.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    ```
2.  **Activate the Environment:**
    - Windows: `.\venv\Scripts\activate`
    - macOS/Linux: `source venv/bin/activate`
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Application:**
    ```bash
    python main.py
    ```

## Configuration
Settings are stored in `config.json`, allowing you to tune system behavior without editing the code.

-   `video_source`: Path to your input video file, or `0` for the default webcam.
-   `grid.*`: Parameters for the matrix grid, such as rows, columns, and cooldown frames.
-   `yolo_*`: Settings for the YOLO detector, like confidence thresholds and how often to run inference.
-   `*speed_threshold`: Speeds at which the system switches between "CITY" and "HIGHWAY" modes, which use different sensitivity thresholds.

video link: "//C:/Users/jomon/MCA/data/video3.mp4"
