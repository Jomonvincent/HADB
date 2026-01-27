import cv2
import time
from input_handler import VideoLoader
from grid_manager import MatrixGrid
from yolo_detector import YoloBrain

def main():
    # 1. Config
    VIDEO_SOURCE = "C:/Users/jomon/MCA/data/video6.mp4"
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    loader = VideoLoader(source=VIDEO_SOURCE, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    grid = MatrixGrid(width=FRAME_WIDTH, height=FRAME_HEIGHT, cooldown_frames=8)
    brain = YoloBrain(conf_threshold=0.5)

    print("Running ADB Hybrid System (YOLO + Brightness)")
    prev_time = 0

    try:
        while True:
            curr_time = time.time()
            frame = loader.get_frame()
            if frame is None: break

            # --- HYBRID PIPELINE ---
            # 1. Run YOLO to get boxes
            vehicle_boxes = brain.detect_vehicles(frame)
            
            # 2. Run Grid Update (Passes frame for brightness + boxes for YOLO)
            # The grid manager now merges them internally
            glare_cells, clean_mask = grid.update(frame, vehicle_boxes)
            
            # 3. Visualize
            output_frame = grid.draw_grid(frame, active_glare_cells=glare_cells)

            # Draw YOLO boxes for debugging
            for box in vehicle_boxes:
                cv2.rectangle(output_frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)

            # FPS
            delta_time = curr_time - prev_time
            prev_time = curr_time
            fps = 1 / delta_time if delta_time > 0 else 0
            cv2.putText(output_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow('ADB Hybrid Output', output_frame)
            cv2.imshow('Debug: Brightness Mask', clean_mask) # See what the brightness logic sees

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        loader.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()