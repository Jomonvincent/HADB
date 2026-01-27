import cv2
import numpy as np

class MatrixGrid:
    def __init__(self, width=640, height=480, rows=8, cols=16, cooldown_frames=5):
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols
        self.cell_w = self.width // self.cols
        self.cell_h = self.height // self.rows
        
        # Cooldown Logic
        self.cooldown_frames = cooldown_frames
        self.cooldown_tracker = np.zeros((rows, cols), dtype=int)

    def get_cell_coordinates(self, row, col):
        x1 = col * self.cell_w
        y1 = row * self.cell_h
        x2 = x1 + self.cell_w
        y2 = y1 + self.cell_h
        return x1, y1, x2, y2

    # --- DETECTOR 1: BRIGHTNESS (Gaussian + Threshold) ---
    def _get_brightness_cells(self, frame, threshold=220, min_blob_area=50):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, raw_mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        
        # Filter noise with contours
        clean_mask = np.zeros_like(raw_mask)
        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_blob_area:
                cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Map to grid
        active_cells = set()
        for row in range(self.rows):
            for col in range(self.cols):
                x1, y1, x2, y2 = self.get_cell_coordinates(row, col)
                cell_roi = clean_mask[y1:y2, x1:x2]
                if cv2.countNonZero(cell_roi) > 0:
                    active_cells.add((row, col))
        
        return active_cells, clean_mask

    # --- DETECTOR 2: YOLO (Bounding Boxes) ---
    def _get_yolo_cells(self, vehicle_boxes):
        active_cells = set()
        for row in range(self.rows):
            for col in range(self.cols):
                cx1, cy1, cx2, cy2 = self.get_cell_coordinates(row, col)
                for box in vehicle_boxes:
                    # --- FIX START ---
                    # We take only the first 4 values (coords) and ignore the label
                    bx1, by1, bx2, by2 = box[:4] 
                    # --- FIX END ---
                    
                    # Intersection logic
                    dx = min(cx2, bx2) - max(cx1, bx1)
                    dy = min(cy2, by2) - max(cy1, by1)
                    if (dx > 0) and (dy > 0):
                        # Trigger if overlap is significant (>10% of cell area)
                        if (dx * dy) > (self.cell_w * self.cell_h * 0.1):
                            active_cells.add((row, col))
                            break 
        return active_cells

    # --- MASTER UPDATE FUNCTION ---
    def update(self, frame, yolo_boxes):
        """
        Combines YOLO and Brightness detection, applies cooldown, 
        and returns the final list of blocked cells.
        """
        # 1. Get inputs from both systems
        brightness_set, debug_mask = self._get_brightness_cells(frame)
        yolo_set = self._get_yolo_cells(yolo_boxes)

        # 2. HYBRID FUSION: Union of both sets (A OR B)
        combined_active_cells = brightness_set.union(yolo_set)

        # 3. Apply Cooldown Logic
        final_blocked_cells = []
        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) in combined_active_cells:
                    # Reset timer if EITHER system detects something
                    self.cooldown_tracker[row, col] = self.cooldown_frames
                else:
                    # Cool down if BOTH are clear
                    if self.cooldown_tracker[row, col] > 0:
                        self.cooldown_tracker[row, col] -= 1
                
                # If timer is running, the cell is blocked
                if self.cooldown_tracker[row, col] > 0:
                    final_blocked_cells.append((row, col))

        return final_blocked_cells, debug_mask

    def draw_grid(self, frame, active_glare_cells=[]):
        # (Same realistic beam drawing logic as Sprint 3)
        overlay = frame.copy()
        beam_color = (255, 255, 0)
        
        for row in range(self.rows):
            for col in range(self.cols):
                x1, y1, x2, y2 = self.get_cell_coordinates(row, col)
                if (row, col) in active_glare_cells:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1) # Blocked
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red Border
                else:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), beam_color, -1) # Light
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), 1)

        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        return frame