import csv
import logging
import os
from datetime import datetime


class GlareLogger:
    """Simple logger to record glare events for post-drive analysis."""

    def __init__(self, log_path=None, level=logging.INFO):
        self.log_path = log_path or os.path.join(os.getcwd(), "glare_events.csv")
        self._ensure_log_file_exists()

        self.logger = logging.getLogger("HADB")
        self.logger.setLevel(level)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def _ensure_log_file_exists(self):
        # Ensure directory exists
        directory = os.path.dirname(self.log_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "trigger", "cell_row", "cell_col", "extra"])

    def log_glare_event(self, trigger: str, cell: tuple, extra: str = ""):
        """Log a glare event to CSV and optionally to stdout."""
        timestamp = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
        row, col = cell
        try:
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, trigger, row, col, extra])
        except Exception as e:
            self.logger.warning(f"Failed to write glare event to log: {e}")

        self.logger.debug(f"Glare event: trigger={trigger} cell={cell} extra={extra}")

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)
