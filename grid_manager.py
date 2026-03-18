import cv2
import numpy as np

class MatrixGrid:
    def __init__(
        self,
        width=640,
        height=480,
        rows=8,
        cols=16,
        cooldown_frames=5,
        safety_buffer_cols=1,
        mask_color=(0, 0, 0),
        legacy_style=True,
        beam_color=(255, 255, 0),
        city_threshold=220,
        highway_threshold=200,
        city_min_blob_area=50,
        highway_min_blob_area=30,
        static_decay=0.02,
        static_threshold=0.6,
        motion_threshold=25,
        motion_required_overlap=5,
        downsample_size=(320, 240),
        roi_top_pct=0.25,
        roi_bottom_pct=0.1,
    ):
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols
        self.cell_w = self.width // self.cols
        self.cell_h = self.height // self.rows

        # Cooldown Logic
        self.cooldown_frames = cooldown_frames
        self.cooldown_tracker = np.zeros((rows, cols), dtype=int)
        # Alpha blending for dimming animation (per cell)
        self.alpha = np.zeros((rows, cols), dtype=float)
        self.target_alpha = np.zeros((rows, cols), dtype=float)
        self.max_alpha = 0.7
        self.alpha_smooth = 0.3
        self.safety_buffer_cols = safety_buffer_cols
        # mask_color should be a tuple (B,G,R) between 0-255
        self.mask_color = tuple(int(c) for c in mask_color)
        # Legacy rendering options (restores previous yellow/red grid look)
        self.legacy_style = bool(legacy_style)
        self.beam_color = tuple(int(c) for c in beam_color)
        self.blocked_border_color = (0, 0, 255)
        self.normal_border_color = (50, 50, 50)

        # Downsample / ROI settings (performance & false-positive reduction)
        self.downsample_size = downsample_size
        self.roi_top_pct = roi_top_pct
        self.roi_bottom_pct = roi_bottom_pct

        # Thresholds & sensitivity (configurable)
        self.city_threshold = city_threshold
        self.highway_threshold = highway_threshold
        self.city_min_blob_area = city_min_blob_area
        self.highway_min_blob_area = highway_min_blob_area

        # Static bright accumulation to ignore fixed streetlights
        # We keep this at the downsampled resolution for performance.
        ds_h, ds_w = self.downsample_size[1], self.downsample_size[0]
        self.static_accum = np.zeros((ds_h, ds_w), dtype=float)
        self.static_mask = np.zeros((ds_h, ds_w), dtype=np.uint8)
        self.static_decay = static_decay  # running average weight for current frame
        self.static_threshold = static_threshold

        # Motion confirmation params
        self.prev_gray = None
        self.motion_threshold = motion_threshold
        self.motion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.motion_required_overlap = motion_required_overlap  # pixels overlap required between contour and motion mask

    def get_cell_coordinates(self, row, col):
        x1 = col * self.cell_w
        y1 = row * self.cell_h
        x2 = x1 + self.cell_w
        y2 = y1 + self.cell_h
        return x1, y1, x2, y2

    # --- DETECTOR 1: BRIGHTNESS (Gaussian + Threshold) ---
    # --- DETECTOR 1: BRIGHTNESS (Gaussian + Threshold) ---
    def _get_brightness_cells(self, frame, threshold=220, min_blob_area=30):
        """Detects bright blobs (headlights) using a downsampled ROI.

        The system defaults to a "safe state" by requiring motion confirmation and
        by ignoring the sky / hood regions.
        """
        # Downsample the frame for faster processing
        ds_frame = cv2.resize(frame, self.downsample_size)
        gray = cv2.cvtColor(ds_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, raw_mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        # Apply vertical ROI: ignore top (sky/streetlights) and bottom (hood)
        h = raw_mask.shape[0]
        top_cut = int(h * self.roi_top_pct)
        bottom_cut = int(h * (1.0 - self.roi_bottom_pct))
        raw_mask[:top_cut, :] = 0
        raw_mask[bottom_cut:, :] = 0

        # Update static accumulation to detect persistent bright pixels (streetlights)
        raw_mask_norm = (raw_mask / 255.0).astype(float)
        self.static_accum = (1.0 - self.static_decay) * self.static_accum + self.static_decay * raw_mask_norm
        self.static_mask = (self.static_accum >= self.static_threshold).astype(np.uint8)

        # Suppress static bright pixels from consideration
        raw_mask_clean = raw_mask.copy()
        raw_mask_clean[self.static_mask == 1] = 0

        # Motion mask (frame-diff) for motion confirmation
        if self.prev_gray is None:
            motion_mask = np.zeros_like(raw_mask)
        else:
            diff = cv2.absdiff(blurred, self.prev_gray)
            _, motion_mask = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self.motion_kernel)

        # Store current gray for next frame
        self.prev_gray = blurred

        # Find contours on cleaned mask
        contours, _ = cv2.findContours(raw_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_blobs_mask = np.zeros_like(raw_mask)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_blob_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = float(h) / float(max(w, 1))

            # Heuristics to ignore tall streetlight-like shapes
            if aspect > 2.0:
                continue

            # Motion confirmation: require some motion overlap unless tracked by YOLO later
            # Count overlap between contour bbox and motion_mask
            roi_motion = motion_mask[y:y+h, x:x+w]
            overlap = int(cv2.countNonZero(roi_motion))
            if overlap < self.motion_required_overlap:
                # small or no motion — treat cautiously, ignore as headlight
                continue

            # Passed heuristics — mark as valid blob
            cv2.drawContours(valid_blobs_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Map to grid using downsampled cell sizes
        active_cells = set()
        ds_cell_w = self.downsample_size[0] // self.cols
        ds_cell_h = self.downsample_size[1] // self.rows

        for row in range(self.rows):
            for col in range(self.cols):
                x1 = col * ds_cell_w
                y1 = row * ds_cell_h
                x2 = x1 + ds_cell_w
                y2 = y1 + ds_cell_h
                cell_roi = valid_blobs_mask[y1:y2, x1:x2]
                if cv2.countNonZero(cell_roi) > 0:
                    active_cells.add((row, col))

        # Build a debug visualization: R=clean raw mask, G=static mask, B=motion mask
        debug_vis_ds = np.zeros((self.downsample_size[1], self.downsample_size[0], 3), dtype=np.uint8)
        debug_vis_ds[:, :, 2] = raw_mask_clean
        debug_vis_ds[:, :, 1] = (self.static_mask * 255).astype(np.uint8)
        debug_vis_ds[:, :, 0] = motion_mask
        debug_vis = cv2.resize(debug_vis_ds, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        return active_cells, debug_vis
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

    def _get_tracked_cells(self, tracked_objects):
        """
        Map tracked centroids (+bbox width) to vertical columns (zones).
        Returns set of (row, col) cells covering the columns across all rows.
        """
        active_cells = set()
        if not tracked_objects:
            return active_cells

        for oid, val in tracked_objects.items():
            try:
                cx, cy, bbox = val
                bx1, by1, bx2, by2 = bbox
                veh_w = bx2 - bx1
                half_w = veh_w / 2.0
                left_px = int(max(0, cx - half_w))
                right_px = int(min(self.width - 1, cx + half_w))
                # apply safety buffer in columns
                left_col = max(0, (left_px // self.cell_w) - self.safety_buffer_cols)
                right_col = min(self.cols - 1, (right_px // self.cell_w) + self.safety_buffer_cols)
                for col in range(left_col, right_col + 1):
                    for row in range(self.rows):
                        active_cells.add((row, col))
            except Exception:
                continue

        return active_cells

    # --- MASTER UPDATE FUNCTION ---
    def update(self, frame, yolo_boxes, tracked_objects=None, mode="CITY", safe_state=False):
        """Update the grid state based on vision inputs.

        Args:
            frame: Current camera frame (BGR).
            yolo_boxes: Latest YOLO detections.
            tracked_objects: Centroid-tracked objects from previous frames.
            mode: "CITY" or "HIGHWAY" (affects sensitivity).
            safe_state: If True, assume sensors are unreliable and apply full dimming.

        Returns:
            final_blocked_cells: list of (row, col) cells that should be dimmed.
            debug_mask: visualization image showing blob/ROI/motion masks.
            source_map: dict mapping each blocked cell to the set of triggers that caused it.
        """
        # Store frame for any visualization / hardware rendering
        self.last_frame = frame

        # If we are in a fail-safe condition, block everything.
        if safe_state or frame is None:
            all_cells = {(r, c) for r in range(self.rows) for c in range(self.cols)}
            # Force cooldown so the UI transitions smoothly
            for r, c in all_cells:
                self.cooldown_tracker[r, c] = self.cooldown_frames
                self.target_alpha[r, c] = self.max_alpha
                self.alpha[r, c] = self.max_alpha
            return list(all_cells), None, {cell: {"SAFE"} for cell in all_cells}

        active_cells = set()
        debug_mask = None

        # 1. Always Run Blob Detection (Primary)
        # We adjust sensitivity based on mode, but we always run it.
        if mode == "CITY":
            blob_cells, debug_mask = self._get_brightness_cells(
                frame,
                threshold=self.city_threshold,
                min_blob_area=self.city_min_blob_area,
            )
        else:  # HIGHWAY
            blob_cells, debug_mask = self._get_brightness_cells(
                frame,
                threshold=self.highway_threshold,
                min_blob_area=self.highway_min_blob_area,
            )

        # 2. Run YOLO (Secondary / Failsafe)
        yolo_cells = self._get_yolo_cells(yolo_boxes)

        # 2b. Map tracked objects (centroid -> columns)
        tracked_cells = self._get_tracked_cells(tracked_objects) if tracked_objects is not None else set()

        # Build a source map for logging
        source_map = {}
        for cell in blob_cells:
            source_map.setdefault(cell, set()).add("BLOB")
        for cell in yolo_cells:
            source_map.setdefault(cell, set()).add("YOLO")
        for cell in tracked_cells:
            source_map.setdefault(cell, set()).add("TRACK")

        # 3. SAFETY UNION: Combine BOTH results
        # If EITHER system sees a threat, we block the cell.
        # Combine blob detector, YOLO and tracked mapping
        combined_active_cells = set(source_map.keys())

        # --- COOLDOWN LOGIC ---
        # Update cooldown and target alpha per cell
        final_blocked_cells = []
        for row in range(self.rows):
            for col in range(self.cols):
                cell = (row, col)
                if cell in combined_active_cells:
                    self.cooldown_tracker[row, col] = self.cooldown_frames
                else:
                    if self.cooldown_tracker[row, col] > 0:
                        self.cooldown_tracker[row, col] -= 1

                # Set target alpha based on whether cell is considered blocked
                if self.cooldown_tracker[row, col] > 0:
                    self.target_alpha[row, col] = self.max_alpha
                else:
                    self.target_alpha[row, col] = 0.0

                # Smooth alpha transition
                self.alpha[row, col] += (self.target_alpha[row, col] - self.alpha[row, col]) * self.alpha_smooth

                # Consider cell blocked if alpha is above small threshold
                if self.alpha[row, col] > 0.03:
                    final_blocked_cells.append(cell)

        return final_blocked_cells, debug_mask, source_map

    def draw_grid(self, frame, active_glare_cells=[]):
        # If legacy style requested, render using previous colored grid (yellow light, red borders)
        if self.legacy_style:
            overlay = frame.copy()
            for row in range(self.rows):
                for col in range(self.cols):
                    x1, y1, x2, y2 = self.get_cell_coordinates(row, col)
                    if (row, col) in active_glare_cells:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1) # Blocked
                        cv2.rectangle(frame, (x1, y1), (x2, y2), self.blocked_border_color, 2)
                    else:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.beam_color, -1) # Light
                        cv2.rectangle(frame, (x1, y1), (x2, y2), self.normal_border_color, 1)

            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            return frame

        # New style: Per-cell alpha blending: shaded overlay on blocked cells, smooth fade
        out = frame

        for row in range(self.rows):
            for col in range(self.cols):
                x1, y1, x2, y2 = self.get_cell_coordinates(row, col)
                a = float(self.alpha[row, col])
                if a <= 0.001:
                    # draw subtle grid boundary only
                    cv2.rectangle(out, (x1, y1), (x2, y2), (40, 40, 40), 1)
                    continue

                # blend a shaded rectangle over ROI using configured mask color
                roi = out[y1:y2, x1:x2]
                mask_rect = np.full_like(roi, self.mask_color)
                blended = cv2.addWeighted(mask_rect, a, roi, 1.0 - a, 0)
                out[y1:y2, x1:x2] = blended
                # draw faint border for blocked zones
                cv2.rectangle(out, (x1, y1), (x2, y2), (20, 20, 20), 1)

        return out