class SpeedSimulator:
    def __init__(self, initial_speed=40):
        """
        Simulates the vehicle's speedometer.
        """
        self.current_speed = initial_speed
        self.mode = "CITY" # Starts in City Mode

    def update(self, key_pressed):
        """
        Adjusts speed based on user input and determines the active logic mode.
        
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
        
        # --- SPRINT 3 LOGIC SWITCH ---
        # Threshold: 60 km/h
        if self.current_speed < 60:
            self.mode = "CITY"
            # In GridManager, this triggers:
            # - HIGH Threshold (220) -> Ignores weak reflections/streetlights
            # - Frame Skipping -> Saves power
        else:
            self.mode = "HIGHWAY"
            # In GridManager, this triggers:
            # - LOW Threshold (200) -> Detects distant headlights early
            # - Max Sensitivity -> Prioritizes safety over false positives
            
        return self.current_speed, self.mode