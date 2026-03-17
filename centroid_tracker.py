import math

class CentroidTracker:
    def __init__(self, max_disappeared=2):
        # objectID -> (cx, cy, (x1,y1,x2,y2))
        self.objects = dict()
        self.disappeared = dict()
        self.next_object_id = 1
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox):
        oid = self.next_object_id
        self.next_object_id += 1
        self.objects[oid] = (centroid[0], centroid[1], bbox)
        self.disappeared[oid] = 0
        return oid

    def deregister(self, oid):
        if oid in self.objects:
            del self.objects[oid]
        if oid in self.disappeared:
            del self.disappeared[oid]

    def update(self, boxes):
        """
        boxes: list of boxes where each box is (x1,y1,x2,y2, ...)
        Returns internal `objects` mapping: objectID -> (cx, cy, bbox)
        """
        input_centroids = []
        input_bboxes = []

        for b in boxes:
            bx1, by1, bx2, by2 = b[:4]
            cx = int((bx1 + bx2) / 2)
            cy = int((by1 + by2) / 2)
            input_centroids.append((cx, cy))
            input_bboxes.append((bx1, by1, bx2, by2))

        # No detections: mark disappeared
        if len(input_centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        # If no existing objects, register all inputs
        if len(self.objects) == 0:
            for i, c in enumerate(input_centroids):
                self.register(c, input_bboxes[i])
            return self.objects

        # Build distance matrix between existing object centroids and new centroids
        object_ids = list(self.objects.keys())
        existing_centroids = [ (self.objects[oid][0], self.objects[oid][1]) for oid in object_ids ]

        D = []
        for ec in existing_centroids:
            row = []
            for ic in input_centroids:
                dx = ec[0] - ic[0]
                dy = ec[1] - ic[1]
                row.append(math.hypot(dx, dy))
            D.append(row)

        # Greedy matching by smallest distances
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
            assigned_rows.add(r)
            assigned_cols.add(c)
            matches.append((r, c))

        unmatched_rows = set(range(len(existing_centroids))) - assigned_rows
        unmatched_cols = set(range(len(input_centroids))) - assigned_cols

        # Update matched objects
        for (r, c) in matches:
            oid = object_ids[r]
            cx, cy = input_centroids[c]
            bbox = input_bboxes[c]
            self.objects[oid] = (cx, cy, bbox)
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

        return self.objects
