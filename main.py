import cv2
import json
import os
import time

from base_hardware import MockHardware, RPiHardware
from centroid_tracker import CentroidTracker
from grid_manager import MatrixGrid
from input_handler import VideoLoader
from speed_controller import SpeedSimulator
from system_logger import GlareLogger
from yolo_detector import YoloBrain


def load_config(config_path="config.json"):
    """Load a JSON config file if it exists (otherwise return empty dict)."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _create_hardware(config, logger):
    hw_type = config.get("hardware", "mock").lower()
    if hw_type == "rpi":
        return RPiHardware(logger=logger)
    return MockHardware(logger=logger)


def main():
    config = load_config()

    logger = GlareLogger(log_path=config.get("log_path"))

    VIDEO_SOURCE = config.get("video_source", "C:/Users/jomon/MCA/data/video3.mp4")
    FRAME_WIDTH = config.get("frame_width", 640)
    FRAME_HEIGHT = config.get("frame_height", 480)
    TARGET_FPS = config.get("target_fps", 30)
    FRAME_DURATION = 1.0 / TARGET_FPS

    YOLO_SKIP_FRAMES = config.get("yolo_skip_frames", 5)

    loader = VideoLoader(source=VIDEO_SOURCE, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    grid = MatrixGrid(width=FRAME_WIDTH, height=FRAME_HEIGHT, **config.get("grid", {}))

    hardware = _create_hardware(config, logger)
    hardware.initialize(config)

    brain = YoloBrain(conf_threshold=config.get("yolo_conf_threshold", 0.5), logger=logger)

    tracker = CentroidTracker(
        max_disappeared=config.get("max_disappeared", 2),
        max_distance_pct=config.get("max_tracker_jump_pct", 0.3),
        frame_width=FRAME_WIDTH,
    )

    car = SpeedSimulator(
        initial_speed=config.get("initial_speed", 40),
        highway_threshold=config.get("highway_speed_threshold", 60),
    )

    logger.info("Running ADB: Smart City Mode (Streetlight Filtering)")

    frame_count = 0
    prev_glare_cells = set()

    try:
        while True:
            start_time = time.time()
            frame = loader.get_frame()
            if frame is None:
                logger.info("End of video stream.")
                break

            frame_count += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            current_speed, mode = car.update(key)

            # Push the latest frame to the YOLO worker (async inference)
            brain.submit_frame(frame)
            vehicle_boxes = brain.get_latest_detections()

            # Update tracker (handles skipped frames and prediction)
            tracked_objects = tracker.update(vehicle_boxes)

            # Determine if we should be in fail-safe (full dimming) state
            safe_state = brain.has_error()

            # Update grid and compute glare/dimming cells
            glare_cells, debug_mask, source_map = grid.update(
                frame,
                vehicle_boxes,
                tracked_objects=tracked_objects,
                mode=mode,
                safe_state=safe_state,
            )

            # Log new glare events for post-drive analysis
            new_cells = set(glare_cells) - prev_glare_cells
            for cell in new_cells:
                triggers = source_map.get(cell, {"UNKNOWN"})
                for trigger in triggers:
                    logger.log_glare_event(trigger, cell, extra=f"mode={mode}")
            prev_glare_cells = set(glare_cells)

            # Apply dimming action to the hardware layer
            hardware.apply_dimming(frame, glare_cells, grid)

            # Build visualization overlay
            output_frame = grid.draw_grid(frame.copy(), active_glare_cells=glare_cells)
            color = (0, 255, 0) if mode == "CITY" else (0, 165, 255)
            cv2.putText(output_frame, f"SPEED: {current_speed} km/h", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(output_frame, f"MODE: {mode}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if debug_mask is not None:
                cv2.imshow('Debug: Filtered Blobs', debug_mask)

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

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        brain.stop()
        hardware.shutdown()
        loader.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()