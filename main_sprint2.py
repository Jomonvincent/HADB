import cv2
import time
from input_handler import VideoLoader
from yolo_detector import YoloBrain

# NOTE: We do NOT import grid_manager here because Sprint 2 is just about Detection.

def main():
    VIDEO_SOURCE = "C:/Users/jomon/MCA/data/video3.mp4" 
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    loader = VideoLoader(source=VIDEO_SOURCE, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    brain = YoloBrain(model_name='yolov8n.pt', conf_threshold=0.5)

    print("--- SPRINT 2: DEEP LEARNING INFERENCE ---")
    print("Goal: Detect Cars, Trucks, and Buses")
    print("Press 'q' to exit.")

    prev_time = 0

    try:
        while True:
            frame = loader.get_frame()
            if frame is None: break

            # 1. Run YOLO Detection
            detections = brain.detect_vehicles(frame)

            # 2. Draw Bounding Boxes (The Goal of Sprint 2)
            for item in detections:
                x1, y1, x2, y2, label = item
                
                # Draw Green Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw Label
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 3. FPS Counter
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Sprint 2: YOLO Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        loader.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()