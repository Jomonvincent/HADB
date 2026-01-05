import cv2
import time
from input_handler import VideoLoader
from grid_manager import MatrixGrid

def main():
    # 1. Configuration
    VIDEO_SOURCE = "C:/Users/jomon/MCA/data/video3.mp4" # Change to 'assets/driving_video.mp4' for a file
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # 2. Initialize Modules
    loader = VideoLoader(source=VIDEO_SOURCE, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    grid = MatrixGrid(width=FRAME_WIDTH, height=FRAME_HEIGHT, rows=8, cols=16)

    print("Starting Adaptive Beam Simulation...")
    print("Press 'q' to exit.")

    try:
        while True:
            # 3. Get Input (The Eye)
            frame = loader.get_frame()

            if frame is None:
                print("Video ended.")
                break

            # 4. Processing (The Logic)
            # (In Sprint 2, we will add the brightness detection here)
            
            # 5. Visualization (The Output)
            # Draw the matrix grid on the current frame
            processed_frame = grid.draw_grid(frame)

            # Display the result
            cv2.imshow('ADB Simulation - Sprint 1', processed_frame)

            # Control framerate (approx 30 FPS)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        # 6. Cleanup
        loader.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    