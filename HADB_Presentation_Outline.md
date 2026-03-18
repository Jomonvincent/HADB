# HADB (Headlight Adaptive Beam) - MCA Main Project Presentation

## Slide 1: Title Slide
**HADB: Hybrid Adaptive Beam System for Automotive Headlights**  
*MCA Main Project*  
*Submitted by: [Your Name]*  
*Date: March 18, 2026*  
*Supervisor: [Supervisor Name]*  

---

## Slide 2: Agenda
- Introduction to Adaptive Driving Beam (ADB)
- Problem Statement & Objectives
- System Architecture
- Technologies & Tools Used
- Key Features & Innovations
- Implementation Details
- Results & Demo
- Challenges & Solutions
- Future Enhancements
- Conclusion

---

## Slide 3: What is Adaptive Driving Beam (ADB)?
- **ADB** automatically adjusts headlight beam pattern to avoid dazzling oncoming drivers
- **Traditional**: Manual high/low beam switching
- **ADB**: Uses sensors to detect vehicles and dynamically dim specific zones
- **Benefits**: Improved safety, reduced glare, better visibility

**Diagram**: Show traditional vs. ADB beam patterns

---

## Slide 4: Problem Statement
- **Examiner's Critique**:
  - Tight Coupling: Code tied to video file, hard to port to real hardware
  - Missing Fail-Safe: No default safe state on sensor failure
  - No External Configuration: Hardcoded thresholds
  - Identity Swapping: Tracker confuses vehicles
  - Linear Logic: Doesn't handle non-linear motion
  - Streetlight False Positives: Bright lights trigger dimming
  - Inconsistent Latency: YOLO every 5 frames causes hiccups
  - Redundant Calculations: Full-frame processing
  - Binary Dimming: On/Off only
  - Lack of Logging: No event recording

---

## Slide 5: Objectives
- **Refactor** for high-level academic & safety standards
- **Hardware Abstraction**: Ready for PC (Mock) or RPi deployment
- **Fail-Safe Mode**: Default to full dimming on failure
- **Advanced Tracking**: Distance thresholding + Kalman-like prediction
- **Vertical ROI**: Ignore sky/streetlights & hood
- **Asynchronous Inference**: YOLO in background thread
- **Downsampling**: Faster blob detection
- **Gradient Dimming**: PWM intensity control
- **System Logging**: CSV event recording

---

## Slide 6: System Architecture
**High-Level Diagram**:
- Input Handler (Video/Camera)
- YOLO Detector (Async Thread)
- Blob Detector (OpenCV)
- Centroid Tracker (with Prediction)
- Grid Manager (Dimming Logic)
- Hardware Layer (Mock/RPi)
- Logger (CSV Events)

**Key Classes**:
- `BaseHardware`: Interface for dimming
- `MockHardware`: PC visualization
- `RPiHardware`: GPIO/PWM stub
- `YoloBrain`: Async YOLO inference
- `MatrixGrid`: Blob + YOLO fusion
- `CentroidTracker`: Motion prediction
- `GlareLogger`: Event logging

---

## Slide 7: Technologies & Tools Used
- **Programming Language**: Python 3.x
- **Computer Vision**: OpenCV (Blob detection, motion masking)
- **Deep Learning**: YOLOv8 Nano (Ultralytics) for vehicle detection
- **Concurrency**: Python threading for async YOLO
- **Configuration**: JSON/YAML for tunable parameters
- **Logging**: Python logging + CSV export
- **Hardware**: GPIO/PWM simulation (RPi.GPIO stub)
- **Development**: VS Code, Git for version control

**Dependencies**: ultralytics, opencv-python, numpy, etc.

---

## Slide 8: Key Features & Innovations
1. **Hardware Abstraction Layer (HAL)**: Seamless switch between PC and embedded
2. **Fail-Safe Logic**: Full dimming on sensor failure (YOLO crash, frame loss)
3. **Advanced Tracking**: Max distance threshold prevents ID swaps; velocity-based prediction
4. **Vertical ROI Filtering**: Ignores top 25% (sky) & bottom 10% (hood)
5. **Asynchronous YOLO**: Background thread maintains 30 FPS
6. **Downsampled Blob Detection**: 320x240 processing for speed
7. **Source-Aware Logging**: Tracks which sensor triggered each dim event
8. **Configurable Parameters**: External JSON for thresholds, paths, etc.

---

## Slide 9: Implementation Details - Code Structure
**Main Components**:
- `main.py`: Orchestrates the loop, handles input/output
- `grid_manager.py`: Hybrid detection (Blob + YOLO), grid logic
- `centroid_tracker.py`: Object tracking with prediction
- `yolo_detector.py`: Async YOLO worker
- `base_hardware.py`: Hardware abstraction
- `system_logger.py`: Event logging
- `config.json`: Tunable parameters

**Key Algorithms**:
- Blob Detection: Gaussian blur, threshold, motion confirmation, static accumulation
- YOLO Integration: Async inference, fallback on error
- Tracking: Distance matrix with max jump, velocity update
- Dimming: Alpha blending for smooth transitions

---

## Slide 10: Results & Demo
**Screenshots**:
- Grid visualization with active dimming zones
- Debug mask showing blob/ROI/motion filters
- FPS counter maintaining 30 FPS
- CSV log example: timestamp, trigger, cell, mode

**Performance Metrics**:
- Maintains target FPS with async YOLO
- Reduces false positives with ROI & static filtering
- Handles sensor failures gracefully
- Logs events for analysis

**Demo Video**: [Link or description of running system]

---

## Slide 11: Challenges & Solutions
- **Challenge**: Tight coupling to video input
  - **Solution**: Hardware abstraction layer
- **Challenge**: Identity swapping in tracking
  - **Solution**: Distance thresholding + prediction
- **Challenge**: YOLO latency hiccups
  - **Solution**: Background threading
- **Challenge**: Streetlight false positives
  - **Solution**: Vertical ROI + static accumulation
- **Challenge**: No logging/debugging
  - **Solution**: CSV logger with source tracking
- **Challenge**: Binary dimming
  - **Solution**: PWM-ready architecture (stub for intensity)

---

## Slide 12: Future Enhancements
- **Real Hardware Integration**: GPIO/PWM on Raspberry Pi
- **Kalman Filter**: Replace simple prediction with full Kalman
- **Gradient Dimming**: Implement 0-255 intensity per cell
- **Multi-Camera Support**: Handle multiple input streams
- **Machine Learning Tuning**: Auto-tune thresholds via ML
- **Real-Time Testing**: On-road validation
- **UI Dashboard**: Web interface for monitoring
- **Edge Deployment**: Optimize for embedded devices

---

## Slide 13: Conclusion
- **HADB** successfully refactored to meet academic & safety standards
- **Hybrid Approach**: Combines YOLO accuracy with OpenCV speed
- **Safety-First**: Fail-safe defaults, logging, and robust tracking
- **Modular Design**: Easy to extend and deploy
- **Impact**: Contributes to safer night driving, reduces accidents from glare

**Thank You!**  
*Questions?*

---

## Slide 14: References
- Ultralytics YOLO Documentation
- OpenCV Tutorials
- Adaptive Driving Beam Standards (SAE J3069)
- Python Threading Guide
- GitHub Repository: [Link to project]

---

## Additional Notes for PPT Creation
- Use professional template with dark background for tech project
- Include diagrams: System block diagram, flowcharts for detection logic
- Embed code snippets where relevant (e.g., key classes)
- Add animations for step-by-step explanations
- Ensure 16:9 aspect ratio for modern displays
- Font: Sans-serif (Arial/Calibri), consistent colors (blue/green theme)