# HADB – Headlight Adaptive Beam System

This project implements an Adaptive Driving Beam (ADB) system using
computer vision techniques.

## Features
- Real-time input handling
- Grid-based beam control
- Modular Python architecture

## Tech Stack
- Python
- OpenCV
- NumPy

## Setup Instructions
1. Create a virtual environment
2. Install dependencies using requirements.txt
3. Run main.py

## Configuration
Settings are stored in `config.json` so you can tune sensitivity and behavior without editing code.
- `video_source` — path to input video (or `0` for webcam)
- `grid.*` — grid size, thresholds, and sensitivity parameters
- `yolo_*` / speed settings — YOLO run frequency, confidence, and mode switch thresholds

