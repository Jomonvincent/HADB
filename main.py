import cv2
from input_handler import VideoLoader
from grid_manager import MatrixGrid

def main():
    # --- 1. CONFIGURATION (Define these variables first) ---
    VIDEO_SOURCE = "C:/Users/jomon/MCA/data/video3.mp4"          # Use 0 for webcam, or path to video file
    FRAME_WIDTH = 640         # <--- These were missing
    FRAME_HEIGHT = 480        # <--- These were missing
    COOLDOWN_FRAMES = 10      # How long (in frames) a cell stays red
    
    # --- 2. INITIALIZE MODULES ---
    loader = VideoLoader(source=VIDEO_SOURCE, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    
    # Initialize grid with the configuration variables
    grid = MatrixGrid(width=FRAME_WIDTH, height=FRAME_HEIGHT, cooldown_frames=COOLDOWN_FRAMES)

    print("Running ADB Simulation (Sprint 2 - Stable)")
    print("Press 'q' to exit.")

    try:
        while True:
            # 3. Get Input
            frame = loader.get_frame()
            if frame is None:
                print("Video ended.")
                break

            # 4. Processing (New Logic with Cooldown)
            # You can tweak 'min_blob_area' here to ignore smaller lights
            glare_cells, clean_mask = grid.scan_for_glare(frame, threshold=220, min_blob_area=20)
            
            # 5. Visualization
            output_frame = grid.draw_grid(frame, active_glare_cells=glare_cells)

            # Show Main Output
            cv2.imshow('ADB Output', output_frame)
            
            # Show Debug Mask (Optional - helps you see what the computer detects)
            cv2.imshow('Debug: Brain View', clean_mask)

            # Exit on 'q'
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        loader.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()