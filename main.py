import cv2
import json
import os
import time
from input_handler import VideoLoader
from grid_manager import MatrixGrid
from centroid_tracker import CentroidTracker
from yolo_detector import YoloBrain
from speed_controller import SpeedSimulator


def load_config(config_path="config.json"):
    """Load a JSON config file if it exists (otherwise return empty dict)."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    config = load_config()

    VIDEO_SOURCE = config.get("video_source", "C:/Users/jomon/MCA/data/video3.mp4")
    FRAME_WIDTH = config.get("frame_width", 640)
    FRAME_HEIGHT = config.get("frame_height", 480)
    TARGET_FPS = config.get("target_fps", 30)
    FRAME_DURATION = 1.0 / TARGET_FPS

    # Run AI every N frames to keep City Mode fast but smart
    YOLO_SKIP_FRAMES = config.get("yolo_skip_frames", 5)

    loader = VideoLoader(source=VIDEO_SOURCE, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    grid = MatrixGrid(
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        **config.get("grid", {}),
    )
    brain = YoloBrain(conf_threshold=config.get("yolo_conf_threshold", 0.5))
    car = SpeedSimulator(
        initial_speed=config.get("initial_speed", 40),
        highway_threshold=config.get("highway_speed_threshold", 60),
    )
    tracker = CentroidTracker(max_disappeared=2)

    print("Running ADB: Smart City Mode (Streetlight Filtering)")
    
    prev_mode = "CITY"
    frame_count = 0
    last_vehicle_boxes = [] 

    try:
        while True:
            start_time = time.time()
            frame = loader.get_frame()
            if frame is None: break
            
            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            
            current_speed, mode = car.update(key)

            # --- KEY CHANGE: ALWAYS RUN YOLO (Periodically) ---
            # We need YOLO in City Mode now to filter streetlights
            if frame_count % YOLO_SKIP_FRAMES == 0:
                last_vehicle_boxes = brain.detect_vehicles(frame)
            
            # Use remembered boxes
            vehicle_boxes = last_vehicle_boxes

            # Update tracker (handles short YOLO flicker)
            tracked_objects = tracker.update(vehicle_boxes)
            # Log tracked positions
            for oid, val in tracked_objects.items():
                cx, cy, bbox = val
                print(f"Tracking Object ID: {oid} at ({cx},{cy})")

            # Pass boxes and tracked objects to Grid Update
            # In CITY mode, these boxes are used to filter blobs (Validation)
            # In HIGHWAY mode, these boxes are used directly (Detection)
            glare_cells, debug_mask = grid.update(frame, vehicle_boxes, tracked_objects=tracked_objects, mode=mode)
            
            # Visualization
            output_frame = grid.draw_grid(frame, active_glare_cells=glare_cells)

            # Draw Overlays
            color = (0, 255, 0) if mode == "CITY" else (0, 165, 255)
            cv2.putText(output_frame, f"SPEED: {current_speed} km/h", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(output_frame, f"MODE: {mode}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show boxes (Optional - to see what YOLO sees)
            for box in vehicle_boxes:
                b = box[:4]
                cv2.rectangle(output_frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)

            # FPS Control
            elapsed_time = time.time() - start_time
            wait_time = FRAME_DURATION - elapsed_time
            if wait_time > 0:
                time.sleep(wait_time)
                fps = TARGET_FPS
            else:
                fps = 1.0 / elapsed_time

            cv2.putText(output_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('ADB Smart Filter', output_frame)
            if mode == "CITY": cv2.imshow('Debug: Filtered Blobs', debug_mask)

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        loader.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()