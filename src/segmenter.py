"""Contour extraction and multi-object tracking."""
import cv2
import numpy as np


class ContourSegmenter:
    def __init__(self, cfg: dict):
        self.min_area = cfg["min_area"]
        self.max_area = cfg["max_area"]

    def find_contours(self, mask: np.ndarray):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filtered = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_area <= area <= self.max_area:
                filtered.append(c)
        return filtered          # <-- THIS LINE WAS MISSING

    def draw_contours(self, frame, contours, color=(0, 255, 0)):
        cv2.drawContours(frame, contours, -1, color, 2)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)