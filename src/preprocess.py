"""Preprocessing: Gaussian filter + adaptive threshold."""
import cv2
import numpy as np


class Preprocessor:
    def __init__(self, cfg: dict):
        self.k = cfg["gaussian_kernel"] | 1        # force odd
        self.sigma = cfg["gaussian_sigma"]
        self.method = cfg["adaptive_method"]
        self.block = cfg["block_size"] | 1
        self.C = cfg["c_constant"]
        self.morph_k = np.ones(
            (cfg["morph_kernel"], cfg["morph_kernel"]), np.uint8
        )

    def to_gray(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self, gray: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(gray, (self.k, self.k), self.sigma)

    def adaptive_threshold(self, blurred: np.ndarray) -> np.ndarray:
        method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if self.method == "gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            method,
            cv2.THRESH_BINARY_INV,
            self.block,
            self.C,
        )
        return thresh

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_k)
        return mask

    def process(self, frame: np.ndarray):
        """Return (gray, blurred, mask)."""
        gray = self.to_gray(frame)
        blurred = self.gaussian_blur(gray)
        mask = self.adaptive_threshold(blurred)
        mask = self.clean_mask(mask)
        return gray, blurred, mask