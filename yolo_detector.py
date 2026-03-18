import threading
import queue
import time

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:
    YOLO = None
    _HAS_ULTRALYTICS = False

class YoloBrain:
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.5, logger=None):
        """Asynchronous YOLO inference worker.

        This class keeps a background thread running so the main loop can keep
        processing frames at a stable frame rate.
        """
        self.logger = logger
        self.conf_threshold = conf_threshold

        if not _HAS_ULTRALYTICS:
            if self.logger:
                self.logger.warning("Ultralytics YOLO not installed. Falling back to dummy detector.")
            self.model = None
            return

        print(f"Loading YOLO Model: {model_name}...")
        self.model = YOLO(model_name)

        # COCO Dataset IDs: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        self.target_classes = [2, 3, 5, 7]
        self.class_names = {
            2: "Car",
            3: "Bike",
            5: "Bus",
            7: "Truck",
        }

        self._frame_queue = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._latest_detections = []
        self._stop_event = threading.Event()
        self._last_error = None

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit_frame(self, frame):
        """Submit a frame for background inference.

        If the background worker is busy, the newest frame replaces the older one.
        """
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                _ = self._frame_queue.get_nowait()
            except Exception:
                pass
            try:
                self._frame_queue.put_nowait(frame)
            except Exception:
                pass

    def get_latest_detections(self):
        """Return the last set of inference detections."""
        with self._result_lock:
            return list(self._latest_detections)

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1)

    def has_error(self):
        return self._last_error is not None

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                detections = self._infer(frame)
                with self._result_lock:
                    self._latest_detections = detections
                    self._last_error = None
            except Exception as e:
                self._last_error = e
                if self.logger:
                    self.logger.error(f"YOLO inference failed: {e}")
                time.sleep(0.1)

    def _infer(self, frame):
        """Run YOLO inference on the given frame."""
        if self.model is None:
            return []

        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = box
                if score > self.conf_threshold and int(class_id) in self.target_classes:
                    c_name = self.class_names.get(int(class_id), "Vehicle")
                    detections.append([int(x1), int(y1), int(x2), int(y2), c_name])

        return detections

    def detect_vehicles(self, frame=None):
        """Legacy API: Return the most recent detections."""
        return self._infer(frame)
        """
        Sprint 2: Inference Loop
        Scans the frame and returns bounding boxes with class names.
        Returns: list of [x1, y1, x2, y2, class_name]
        """
        results = self.model(frame, verbose=False) # Run inference
        detections = []

        for result in results:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = box
                
                # Filter: Only accept if confidence is high AND it is a vehicle
                if score > self.conf_threshold and int(class_id) in self.target_classes:
                    c_name = self.class_names.get(int(class_id), "Vehicle")
                    # Return box coordinates + label
                    detections.append([int(x1), int(y1), int(x2), int(y2), c_name])
        
        return detections