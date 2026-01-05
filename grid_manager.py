import cv2

class MatrixGrid:
    def __init__(self, width=640, height=480, rows=8, cols=16):
        """
        Initializes the LED Matrix Grid.
        """
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols
        
        # Calculate the size of each individual cell (LED block)
        self.cell_w = self.width // self.cols
        self.cell_h = self.height // self.rows

    def draw_grid(self, frame):
        """
        Draws the grid lines on the given frame.
        """
        # Draw Horizontal Lines
        for i in range(1, self.rows):
            y = i * self.cell_h
            cv2.line(frame, (0, y), (self.width, y), (0, 255, 0), 1)

        # Draw Vertical Lines
        for j in range(1, self.cols):
            x = j * self.cell_w
            cv2.line(frame, (x, 0), (x, self.height), (0, 255, 0), 1)
            
        return frame

    def get_cell_coordinates(self, row, col):
        """
        Returns the (x1, y1, x2, y2) bounding box for a specific cell.
        Useful later for detection logic.
        """
        x1 = col * self.cell_w
        y1 = row * self.cell_h
        x2 = x1 + self.cell_w
        y2 = y1 + self.cell_h
        return x1, y1, x2, y2