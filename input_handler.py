import cv2

class VideoLoader:
    def __init__(self, source=0, width=640, height=480):
        """
        Initializes the video loader.
        
        Args:
            source: Path to a video file (e.g., 'assets/driving.mp4') or 0 for webcam.
            width (int): Target width for resizing.
            height (int): Target height for resizing.
        """
        self.cap = cv2.VideoCapture(source)
        self.width = width
        self.height = height
        
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video source '{source}'")

    def get_frame(self):
        """
        Reads a frame from the video source and resizes it.
        
        Returns:
            frame: The resized video frame, or None if the video has ended.
        """
        ret, frame = self.cap.read()
        
        if not ret:
            return None  # End of video or error
        
        # Frame Pre-processing: Resize to standard resolution
        # This ensures consistent processing speed regardless of input video size
        resized_frame = cv2.resize(frame, (self.width, self.height))
        
        return resized_frame

    def release(self):
        """Releases the video capture resource."""
        self.cap.release()

# --- Testing the Module (Run this file directly to test) ---
if __name__ == "__main__":
    # Usage: Replace 0 with 'path/to/your/video.mp4' to test a file
    # 0 uses the default webcam
    loader = VideoLoader(source=0) 

    print("Press 'q' to exit the video window.")

    while True:
        frame = loader.get_frame()

        if frame is None:
            print("End of video stream.")
            break

        # Display the resulting frame
        cv2.imshow('Input Feed (Resized)', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    loader.release()
    cv2.destroyAllWindows()