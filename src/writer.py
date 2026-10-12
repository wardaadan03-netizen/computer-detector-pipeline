"""Video and CSV output writers for the CV pipeline."""
import csv
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class VideoWriterWrapper:
    """Thin wrapper around cv2.VideoWriter with safe release and path handling."""

    def __init__(self, path: str, fps: float, frame_size: tuple[int, int],
                 codec: str = "mp4v", is_color: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.path), fourcc, fps, frame_size, isColor=is_color
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {self.path}")

        self.frame_count = 0

    def write(self, frame: np.ndarray) -> None:
        # Guarantee correct color space for mask output
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        self.writer.write(frame)
        self.frame_count += 1

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class CSVLogger:
    """Streaming CSV logger that writes one row per frame."""

    FIELDS = [
        "frame", "timestamp_s", "latency_ms", "fps",
        "detections", "contours", "mean_conf",
        "classes_json", "resolution_w", "resolution_h",
    ]

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def log(self, row: dict) -> None:
        # Fill missing keys with defaults
        safe_row = {k: row.get(k, "") for k in self.FIELDS}
        self._writer.writerow(safe_row)
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SummaryWriter:
    """Persist aggregated run statistics as JSON."""

    @staticmethod
    def write(path: str, summary: dict) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)