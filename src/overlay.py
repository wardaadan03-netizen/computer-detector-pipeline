"""Dynamic overlay drawing: class counts, FPS, boxes."""
import cv2
import numpy as np
from collections import Counter


class Overlay:
    def __init__(self, cfg: dict):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.scale = cfg["font_scale"]
        self.thickness = cfg["thickness"]
        self.box_color = tuple(cfg["box_color"])
        self.text_color = tuple(cfg["text_color"])
        self.show_fps = cfg["show_fps"]
        self.show_counts = cfg["show_counts"]

    def draw_detections(self, frame, detections):
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = f'{d["cls_name"]} {d["conf"]:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, 2)

            (tw, th), _ = cv2.getTextSize(
                label, self.font, self.scale, self.thickness
            )
            cv2.rectangle(frame,
                          (x1, y1 - th - 8),
                          (x1 + tw + 6, y1),
                          self.box_color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        self.font, self.scale, (0, 0, 0), self.thickness)

    def draw_counts(self, frame, detections):
        counts = Counter(d["cls_name"] for d in detections)
        y = 60
        cv2.putText(frame, f"Objects: {sum(counts.values())}",
                    (10, y), self.font, 0.7, self.text_color, 2)
        y += 25
        for cls_name, n in counts.items():
            cv2.putText(frame, f"  {cls_name}: {n}",
                        (10, y), self.font, 0.6, self.text_color, 2)
            y += 22

    def draw_fps(self, frame, fps):
        if self.show_fps:
            cv2.putText(frame, f"FPS: {fps:.2f}",
                        (frame.shape[1] - 140, 30),
                        self.font, 0.7, (0, 255, 255), 2)