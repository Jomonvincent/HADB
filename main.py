import cv2
from input_handler import VideoLoader
from grid_manager import MatrixGrid

def main():
    VIDEO_SOURCE = "C:/Users/jomon/MCA/data/video3.mp4" # Or 'path/to/video.mp4'
    loader = VideoLoader(source=VIDEO_SOURCE)
    grid = MatrixGrid()

    print("Running ADB Simulation (Sprint 1 Completed)")
    
    while True:
        frame = loader.get_frame()
        if frame is None: break

        # --- NEW LOGIC HERE ---
        # 1. Analyze the frame for glare
        glare_cells, binary_mask = grid.scan_for_glare(frame, threshold=220)
        
        # 2. Draw the grid, passing the detected glare cells
        output_frame = grid.draw_grid(frame, active_glare_cells=glare_cells)
        # ----------------------

        cv2.imshow('ADB Output', output_frame)
        
        # Optional: Show the "Computer Vision View" (The black and white mask)
        # This helps you debug if the threshold (220) is too high or too low
        cv2.imshow('Debug: Threshold Mask', binary_mask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    loader.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()