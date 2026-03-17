class SpeedSimulator:
    def __init__(self, initial_speed=40, highway_threshold=60):
        """Simulates the vehicle's speedometer.

        Args:
            initial_speed (int): Starting speed in km/h.
            highway_threshold (int): Speed at which mode switches from CITY to HIGHWAY.
        """
        self.current_speed = initial_speed
        self.mode = "CITY"  # Starts in City Mode
        self.highway_threshold = highway_threshold

    def update(self, key_pressed):
        """Adjusts speed based on user input and determines the active logic mode.

        Args:
            key_pressed: The ASCII key code from cv2.waitKey()

        Returns:
            tuple: (current_speed, mode_string)
        """
        # Controls: 'W' to Accelerate, 'S' to Decelerate
        # We use increments of 2 to make the simulation feel responsive
        if key_pressed == ord('w') or key_pressed == ord('W'):
            self.current_speed += 2
        elif key_pressed == ord('s') or key_pressed == ord('S'):
            self.current_speed -= 2

        # Clamp speed to realistic limits (0 to 140 km/h)
        self.current_speed = max(0, min(self.current_speed, 140))

        # --- MODE SWITCH ---
        # Switch modes based on configured speed threshold
        if self.current_speed < self.highway_threshold:
            self.mode = "CITY"
        else:
            self.mode = "HIGHWAY"

        return self.current_speed, self.mode
