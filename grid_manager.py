import cv2
import numpy as np # Needed for handling pixel arrays

class MatrixGrid:
    def __init__(self, width=640, height=480, rows=8, cols=16):
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols
        self.cell_w = self.width // self.cols
        self.cell_h = self.height // self.rows

    def get_cell_coordinates(self, row, col):
        """Helper to get bounding box of a specific cell."""
        x1 = col * self.cell_w
        y1 = row * self.cell_h
        x2 = x1 + self.cell_w
        y2 = y1 + self.cell_h
        return x1, y1, x2, y2

    def scan_for_glare(self, frame, threshold=220):
        """
        TASK 4: LOGIC IMPLEMENTATION
        1. Convert to Grayscale
        2. Threshold to find bright spots
        3. Map spots to grid cells
        """
        # [Step 1] Grayscale Conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # [Step 2] Bright Spot Thresholding
        # Pixels brighter than 'threshold' become 255 (White), others 0 (Black)
        # Note: 220 is a high threshold (only very bright lights), lower it if detection fails.
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        glare_cells = [] # List to store (row, col) of cells detecting glare

        # [Step 3] Pixel-to-Grid Mapping
        for row in range(self.rows):
            for col in range(self.cols):
                # Get the boundaries for this specific LED cell
                x1, y1, x2, y2 = self.get_cell_coordinates(row, col)

                # Extract just this cell's area from the threshold mask
                cell_roi = mask[y1:y2, x1:x2]

                # Count how many "bright" pixels are in this cell
                bright_pixel_count = cv2.countNonZero(cell_roi)

                # If we see enough bright pixels (sensitivity), mark this cell
                if bright_pixel_count > 5:  
                    glare_cells.append((row, col))

        return glare_cells, mask

    def draw_grid(self, frame, active_glare_cells=[]):
        """
        Draws the grid. If a cell is in 'active_glare_cells', it draws it RED (blocked).
        Otherwise, it draws it GREEN (safe/illuminated).
        """
        # Draw the logic results
        for row in range(self.rows):
            for col in range(self.cols):
                x1, y1, x2, y2 = self.get_cell_coordinates(row, col)
                
                if (row, col) in active_glare_cells:
                    # GLARE DETECTED: Draw filled Red rectangle (Simulating LED OFF/Masking)
                    # Use alpha blending for a transparent look if desired, but simple rect for now
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red Border
                    cv2.putText(frame, "OFF", (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
                else:
                    # SAFE: Draw Green Border (Simulating LED ON)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        return frame