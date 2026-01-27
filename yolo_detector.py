from ultralytics import YOLO

class YoloBrain:
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.5):
        """
        Sprint 2: Deep Learning Core
        Initializes the YOLOv8 Nano model.
        """
        print(f"Loading YOLO Model: {model_name}...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        
        # Sprint 2: Class Filtering
        # COCO Dataset IDs: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        self.target_classes = [2, 3, 5, 7] 
        
        # Mapping IDs to Names for display
        self.class_names = {
            2: "Car",
            3: "Bike",
            5: "Bus",
            7: "Truck"
        }

    def detect_vehicles(self, frame):
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