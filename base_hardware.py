import abc
import cv2


class BaseHardware(abc.ABC):
    """Hardware abstraction layer for Headlight Adaptive Beam (HADB).

    This interface allows the core algorithm to remain unchanged when switching
    between a PC simulation (MockHardware) and a real embedded target (Raspberry Pi).
    """

    @abc.abstractmethod
    def initialize(self, config: dict):
        """Initialize hardware resources based on configuration."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_dimming(self, frame, active_cells, grid):
        """Apply the computed dimming state to the hardware.

        Args:
            frame: The current camera frame that can be used for visualization.
            active_cells (set of (row, col)): Cells that should be dimmed (blocked).
            grid (MatrixGrid): The grid manager (used for visualization in mock mode).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def shutdown(self):
        """Cleanly shut down hardware resources."""
        raise NotImplementedError


class MockHardware(BaseHardware):
    """A software-only hardware implementation for local testing.

    This class uses the MatrixGrid drawing logic to visualize what would have
    been sent to a real hardware dimmer.
    """

    def __init__(self, window_name="HADB Mock Hardware", logger=None):
        self.window_name = window_name
        self.logger = logger
        self.last_active_cells = set()
        self.initialized = False

    def initialize(self, config: dict):
        self.initialized = True
        # No real hardware to initialize
        if self.logger:
            self.logger.info("MockHardware initialized")

    def apply_dimming(self, frame, active_cells, grid):
        """Render a visualization of the dimming grid.

        The grid itself is responsible for alpha smoothing so we can keep the
        same visual behavior as the embedded target.
        """
        if not self.initialized:
            raise RuntimeError("Hardware not initialized")

        prev_cells = set(self.last_active_cells)
        self.last_active_cells = set(active_cells)

        # Draw the grid overlay and show the window
        out_frame = grid.draw_grid(frame.copy(), active_glare_cells=active_cells)
        cv2.imshow(self.window_name, out_frame)

        # Log any new glare cells (for debugging)
        if self.logger:
            added = set(active_cells) - prev_cells
            for cell in added:
                self.logger.debug(f"MockHardware dimming cell {cell}")

    def shutdown(self):
        if self.logger:
            self.logger.info("MockHardware shutdown")
        # Close OpenCV window if present
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass


class RPiHardware(BaseHardware):
    """Placeholder for Raspberry Pi hardware integration.

    This class should be extended with real GPIO control logic for the
    target hardware (e.g., PWM channels for each grid column/row).
    """

    def __init__(self, gpio_module=None, logger=None):
        self.gpio = gpio_module
        self.logger = logger
        self.pwms = {}
        self.initialized = False

    def initialize(self, config: dict):
        # Example stub: set up GPIO mode and PWM channels
        # This file intentionally does not import RPi.GPIO directly to keep
        # the desktop environment working.
        self.initialized = True
        if self.logger:
            self.logger.info("RPiHardware initialized (stub)")

    def apply_dimming(self,frame, active_cells, grid):
        if not self.initialized:
            raise RuntimeError("Hardware not initialized")
        # TODO: Convert active_cells into PWM output values. Example:
        # pwm_value = 255 if (row, col) in active_cells else 0
        # pwm_channel.ChangeDutyCycle(pwm_value / 255.0 * 100)
        if self.logger:
            self.logger.debug(f"RPiHardware would dim {len(active_cells)} cells")

    def shutdown(self):
        if self.logger:
            self.logger.info("RPiHardware shutdown (stub)")
        # TODO: Cleanup GPIO
