import math


class CentroidTracker:
    def __init__(self, max_disappeared=2, max_distance_pct=0.3, frame_width=640):
        """Tracks object centroids with a simple motion model.

        Args:
            max_disappeared: Frames allowed with missing detections before deregistering.
            max_distance_pct: If an object jumps more than this fraction of the frame width,
                it is treated as a new detection rather than a match.
            frame_width: Used to compute the max jump distance.
        """
        self.objects = {}  # object_id -> { centroid, bbox, velocity }
        self.disappeared = {}  # object_id -> frames missing
        self.next_object_id = 1
        self.max_disappeared = max_disappeared
        self.max_distance_pct = max_distance_pct
        self.max_distance = int(frame_width * max_distance_pct)

    def _predict_centroid(self, obj):
        cx, cy = obj["centroid"]
        vx, vy = obj.get("velocity", (0, 0))
        return (int(cx + vx), int(cy + vy))

    def register(self, centroid, bbox):
        oid = self.next_object_id
        self.next_object_id += 1
        self.objects[oid] = {
            "centroid": (centroid[0], centroid[1]),
            "bbox": bbox,
            "velocity": (0, 0),
        }
        self.disappeared[oid] = 0
        return oid

    def deregister(self, oid):
        if oid in self.objects:
            del self.objects[oid]
        if oid in self.disappeared:
            del self.disappeared[oid]

    def update(self, boxes):
        """Update tracked objects based on newly detected bounding boxes.

        Args:
            boxes: list of boxes in the form (x1, y1, x2, y2, ...).

        Returns:
            dict: objectID -> (cx, cy, bbox)
        """
        input_centroids = []
        input_bboxes = []

        if boxes is None:
            # Sensor failure / missing frame: keep predictions but do not update.
            for oid, obj in list(self.objects.items()):
                predicted = self._predict_centroid(obj)
                # Keep bbox unchanged since we don't have current detection
                self.objects[oid]["centroid"] = predicted
            return {oid: (obj["centroid"][0], obj["centroid"][1], obj["bbox"]) for oid, obj in self.objects.items()}

        for b in boxes:
            bx1, by1, bx2, by2 = b[:4]
            cx = int((bx1 + bx2) / 2)
            cy = int((by1 + by2) / 2)
            input_centroids.append((cx, cy))
            input_bboxes.append((bx1, by1, bx2, by2))

        # No detections: use predicted positions (skip frame prediction)
        if len(input_centroids) == 0:
            for oid, obj in list(self.objects.items()):
                predicted = self._predict_centroid(obj)
                self.objects[oid]["centroid"] = predicted
                # Do not mark disappeared during inference skip frames
            return {oid: (obj["centroid"][0], obj["centroid"][1], obj["bbox"]) for oid, obj in self.objects.items()}

        # If no existing objects, register all inputs
        if len(self.objects) == 0:
            for i, c in enumerate(input_centroids):
                self.register(c, input_bboxes[i])
            return {oid: (obj["centroid"][0], obj["centroid"][1], obj["bbox"]) for oid, obj in self.objects.items()}

        # Build distance matrix between predicted existing centroids and new centroids
        object_ids = list(self.objects.keys())
        predicted_centroids = [self._predict_centroid(self.objects[oid]) for oid in object_ids]

        D = []
        for pc in predicted_centroids:
            row = []
            for ic in input_centroids:
                dx = pc[0] - ic[0]
                dy = pc[1] - ic[1]
                row.append(math.hypot(dx, dy))
            D.append(row)

        # Greedy matching by smallest distances (with max distance threshold)
        assigned_rows = set()
        assigned_cols = set()
        matches = []  # (rowIdx, colIdx)
        distance_items = []
        for r in range(len(D)):
            for c in range(len(D[0])):
                distance_items.append((D[r][c], r, c))
        distance_items.sort(key=lambda x: x[0])

        for dist, r, c in distance_items:
            if r in assigned_rows or c in assigned_cols:
                continue
            if dist > self.max_distance:
                # Too far to be a plausible match; treat as new object
                continue
            assigned_rows.add(r)
            assigned_cols.add(c)
            matches.append((r, c))

        unmatched_rows = set(range(len(predicted_centroids))) - assigned_rows
        unmatched_cols = set(range(len(input_centroids))) - assigned_cols

        # Update matched objects
        for (r, c) in matches:
            oid = object_ids[r]
            cx, cy = input_centroids[c]
            bbox = input_bboxes[c]
            old_centroid = self.objects[oid]["centroid"]
            vx = cx - old_centroid[0]
            vy = cy - old_centroid[1]
            self.objects[oid]["centroid"] = (cx, cy)
            self.objects[oid]["bbox"] = bbox
            self.objects[oid]["velocity"] = (vx, vy)
            self.disappeared[oid] = 0

        # Increase disappeared for unmatched existing objects
        for r in unmatched_rows:
            oid = object_ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        # Register new input detections that didn't match existing objects
        for c in unmatched_cols:
            self.register(input_centroids[c], input_bboxes[c])

        return {oid: (obj["centroid"][0], obj["centroid"][1], obj["bbox"]) for oid, obj in self.objects.items()}
