import cv2
import numpy as np

class MatrixGrid:
    def __init__(self, width=640, height=480, rows=8, cols=16, cooldown_frames=5):
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols
        
        # Grid dimensions
        self.cell_w = self.width // self.cols
        self.cell_h = self.height // self.rows

        # --- NEW: Temporal Smoothing Variables ---
        self.cooldown_frames = cooldown_frames
        # A 2D array to track how long each cell must stay "OFF"
        # If value > 0, the cell is blocked. If 0, it is safe.
        self.cooldown_tracker = np.zeros((rows, cols), dtype=int)

    def get_cell_coordinates(self, row, col):
        x1 = col * self.cell_w
        y1 = row * self.cell_h
        x2 = x1 + self.cell_w
        y2 = y1 + self.cell_h
        return x1, y1, x2, y2

    def scan_for_glare(self, frame, threshold=220, min_blob_area=20):
        """
        Detects glare and updates the cooldown timers.
        Returns:
            final_blocked_cells: List of cells that should be visually RED.
            clean_mask: For debugging.
        """
        # 1. Image Pre-processing (Blurring reduces random noise)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Gaussian Blur
        _, raw_mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        # 2. Contour Filtering (From previous task)
        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(raw_mask)

        for cnt in contours:
            if cv2.contourArea(cnt) > min_blob_area:
                cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # 3. Update Cooldown Logic
        current_frame_glare = set()

        # Identify which cells have glare RIGHT NOW
        for row in range(self.rows):
            for col in range(self.cols):
                x1, y1, x2, y2 = self.get_cell_coordinates(row, col)
                cell_roi = clean_mask[y1:y2, x1:x2]
                
                if cv2.countNonZero(cell_roi) > 0:
                    current_frame_glare.add((row, col))

        # Update the Timers
        final_blocked_cells = []
        
        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) in current_frame_glare:
                    # CASE A: Glare Detected NOW
                    # Reset timer to max (keep it blocked)
                    self.cooldown_tracker[row, col] = self.cooldown_frames
                else:
                    # CASE B: No Glare Now
                    # Decrease timer (cool down)
                    if self.cooldown_tracker[row, col] > 0:
                        self.cooldown_tracker[row, col] -= 1

                # If the timer is still active, the cell is considered BLOCKED
                if self.cooldown_tracker[row, col] > 0:
                    final_blocked_cells.append((row, col))

        return final_blocked_cells, clean_mask

    def draw_grid(self, frame, active_glare_cells=[]):
        for row in range(self.rows):
            for col in range(self.cols):
                x1, y1, x2, y2 = self.get_cell_coordinates(row, col)
                
                if (row, col) in active_glare_cells:
                    # RED (Blocked/OFF)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # Optional: Show the timer value to see the cooldown working
                    timer_val = self.cooldown_tracker[row, col]
                    cv2.putText(frame, str(timer_val), (x1+10, y1+30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                else:
                    # GREEN (Safe/ON)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        return frame