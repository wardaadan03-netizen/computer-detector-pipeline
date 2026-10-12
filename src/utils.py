"""Utility helpers for the CV pipeline."""
import cv2
import numpy as np
from pathlib import Path
import yaml

# Project root = parent of the "src" directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """
    Load YAML config. If path is None, loads configs/default.yaml
    relative to the project root (not the current working directory).
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG

    # If a relative path was passed, resolve it from the project root
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {cfg_path}\n"
            f"Project root detected as: {PROJECT_ROOT}\n"
            f"Make sure configs/default.yaml exists."
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> None:
    """Create parent directory if it doesn't exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def resize_keep_aspect(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)